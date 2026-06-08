import asyncio
from datetime import datetime
import json
import os
import platform
import threading
import time
from typing import Dict, List, Literal, Optional
from dataclasses import dataclass

import httpx
from pydantic import BaseModel

import litellm
from litellm._logging import verbose_logger
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.integrations.custom_batch_logger import CustomBatchLogger
from litellm.llms.watsonx.common_utils import IBMWatsonXMixin, _get_api_params
from litellm.secret_managers.main import get_secret_bool, get_secret_str
from litellm.types.integrations.watsonx_governance import WOSDatasetInfo, WOSPayloadLoggingObject, PromptAssetDetails, WOSSubscriptionInfoDict
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.types.llms.watsonx import WatsonXAPIParams

DEFAULT_LITELLM_PROMPT_TEMPLATE = "{chat_history}\n{model}:"
wxgov_cache = InMemoryCache()

class WatsonXGovernanceLogger( CustomBatchLogger):

    _metadata_suffix = "watsonx_gov"

    def __init__(self, **kwargs):

        self._stream_id_to_span = {}
        self._lock = threading.Lock()  # lock for _stream_id_to_span
        # stores the subscription details for a container
        batch_size = kwargs.pop("batch_size", None) or os.getenv("WATSONX_GOV_BATCH_SIZE", None)
        if batch_size is not None:
            batch_size = int(batch_size)
        self.log_queue = []
        # log the payload logging data every flush_interval seconds
        asyncio.create_task(self.periodic_flush())
        self.flush_lock = asyncio.Lock()
        self.aclient = litellm.module_level_aclient
        self.client = litellm.module_level_client
        super().__init__(**kwargs, flush_lock=self.flush_lock, batch_size=batch_size)

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        try:
            with open("litellm_wxgov.log", "a") as f:
                f.write(
                    f"Success Event - Start Time: {start_time}, End Time: {end_time} - Response: {response_obj.model_dump_json(exclude_none=True)} - Params: {json.dumps({k:v for k,v in kwargs.items() if v is not None}, default=str)}\n"
                )
            with open("litellm_wxgov_logs.jsonl", "a") as f:
                f.write(
                    json.dumps({
                        "event": "success",
                        "start_time": str(start_time),
                        "end_time": str(end_time),
                        "response": response_obj.model_dump(exclude_none=True),
                        "kwargs": {k:v for k,v in kwargs.items() if v is not None},
                    }, default=str) + "\n"
                )
            if kwargs.get("stream") and not kwargs.get("complete_streaming_response"):
                # If stream is enabled but not complete, we don't log the success event yet
                return
            # get the details of the subscription for this ai asset / prompt template
            asyncio.create_task(self.async_log_success_event(kwargs, response_obj, start_time, end_time))
        except Exception as err:
            with open("litellm_wxgov.log", "a") as f:
                f.write(
                    f"Error logging success event - Start Time: {start_time}, End Time: {end_time} - Error: {str(err)}\n"
                )
            verbose_logger.exception("watsonx.governance logging Error", stack_info=True)

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        subscription_details = await self.aget_subscription_details(kwargs)
        payload_log_data = self._prepare_payload_data(
            subscription_details,
            response_obj,
            start_time,
            end_time,
            kwargs,
        )
        self.log_queue.append(payload_log_data)
        if len(self.log_queue) >= self.batch_size:
            verbose_logger.debug(
                "[watsonx.governance] Flushing payload logging data to watsonx.governance OpenScale."
            )
            self._send_batch()
        else:
            verbose_logger.debug(
                f"[watsonx.governance] Added payload logging data to queue. It will flush in {self.flush_interval} seconds."
            )

    async def aget_subscription_details(self, kwargs) -> WOSSubscriptionInfoDict:
        """
        Get the subscription details for the current container.
        This will return the subscription details for the current container.
        If the subscription details are not found, it will raise an error.
        """
        wx_mixin = IBMWatsonXMixin()
        base_url = wx_mixin._get_base_url(kwargs.pop("api_base", None))
        headers = wx_mixin.validate_environment(
            headers={},
            model="",
            messages=[],
            api_key=kwargs.pop("api_key", None),
            api_base=base_url,
            litellm_params=kwargs.pop("litellm_params", {}),
            optional_params=kwargs,
        )
        wos_url = _get_watson_openscale_url(
            base_url=base_url, 
            headers=headers, 
            **kwargs
        )
        container_type = "project" if kwargs.get("project_id") else "space"
        resp = await self.aclient.get(
            f"{wos_url.rstrip('/')}/v2/subscriptions",
            headers=headers,
            params={
                "project_id": kwargs.get("project_id"),
                "space_id": kwargs.get("space_id"),
                "deployment_id": kwargs.get("deployment_id"),
                "prompt_template_asset_id": kwargs.get("prompt_template_asset_id"),
                "version": kwargs.get("version", litellm.WATSONX_DEFAULT_API_VERSION),
            },
        )
        dict_resp = resp.json()
        subscriptions = dict_resp.get("resources", [])
        if not subscriptions:
            raise ValueError(
                f"No subscription found for the current container. Please create a subscription in watsonx.governance OpenScale for the current container. URL: {wos_url.rstrip('/')}/v2/subscriptions"
            )
        subscription: WOSSubscriptionInfoDict = subscriptions[0].get("entity", {})
        verbose_logger.debug(
            f"[watsonx.governance] Found subscription for the current container: {subscription.get('subscription_id')}"
        )
        return subscription

    def _send_batch(self):
        """Calls async_send_batch in an event loop"""
        if not self.log_queue:
            return

        try:
            # Try to get the existing event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're already in an event loop, create a task
                asyncio.create_task(self.async_send_batch())
            else:
                # If no event loop is running, run the coroutine directly
                loop.run_until_complete(self.async_send_batch())
        except RuntimeError:
            # If we can't get an event loop, create a new one
            asyncio.run(self.async_send_batch())

    async def async_send_batch(self, **kwargs):
        if not self.log_queue:
            # no payload logging data to send
            return
        # send the batch to Watson OpenScale
        try:
            verbose_logger.debug(
                "[watsonx.governance] Sending payload logging data to Watson OpenScale."
            )
            pl_batches = self._group_payloads_by_dataset()
            for dataset_id, batch in pl_batches.items():
                await self._log_batch_to_watson_openscale(dataset_id, batch)
        except Exception:
            verbose_logger.exception("Error sending payload logging data", stack_info=True)
    
    def _get_wos_url_and_headers(self, kwargs):
        wx_mixin = IBMWatsonXMixin()
        base_url = wx_mixin._get_base_url(kwargs.pop("api_base", None))
        headers = wx_mixin.validate_environment(
            headers={}, model="", messages=[],
            api_key=kwargs.pop("api_key", None),
            api_base=kwargs.pop("api_base", base_url),
            litellm_params=kwargs.pop("litellm_params", {}),
            optional_params=kwargs,
        )
        wos_url = _get_watson_openscale_url(
            base_url=base_url, 
            headers=headers, 
            **kwargs
        )
        return wos_url, headers

    def _group_payloads_by_dataset(self) -> Dict[str, List[WOSPayloadLoggingObject]]:
        """
        Group the payload logging data by subscription.
        """
        pl_batches:Dict[str, List[WOSPayloadLoggingObject]] = {}
        for pl_data in self.log_queue:
            if pl_data["dataset_id"] not in pl_batches:
                pl_batches[pl_data["dataset_id"]] = []
            pl_batches[pl_data["dataset_id"]].append(pl_data)
        return pl_batches
    
    async def _log_batch_to_watson_openscale(self, dataset_id: str, pl_data: List[WOSPayloadLoggingObject], **kwargs):
        """
        Log the payload logging data to Watson OpenScale.
        """
        if not pl_data:
            return
        try:
            wos_url, headers = self._get_wos_url_and_headers(kwargs)
            # create the payload logging object
            avg_resp_time = round(sum([pl["response_time"] for pl in pl_data])/len(pl_data),3)
            payload = {
                "request": {
                    "fields": pl_data[0]["request"]["fields"],
                    "values": [pl["request"]["values"] for pl in pl_data],
                },
                "response": {
                    "fields": pl_data[0]["response"]["fields"],
                    "values": [pl["response"]["values"] for pl in pl_data],
                },
                "response_time": avg_resp_time,
            }
            url = f"{wos_url.rstrip('/')}/v2/data_sets/{dataset_id}/records"
            verbose_logger.debug(f"[watsonx.governance] Sending payload logging data with url: {url}")
            resp = await self.aclient.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            if resp.status_code >= 300:
                verbose_logger.error(
                    f"[watsonx.governance] Error sending payload logging data to Watson OpenScale: {resp.status_code} {resp.text}"
                )
            else:
                verbose_logger.debug(
                    "[watsonx.governance] Payload logging data sent to Watson OpenScale successfully."
                )
        except httpx.HTTPStatusError as e:
            verbose_logger.exception(
                f"[watsonx.governance] Error sending payload logging data to Watson OpenScale: {e.response.status_code} {e.response.text}",
            )
        except Exception as e:
            verbose_logger.exception(
                f"[watsonx.governance] Error sending payload logging data to Watson OpenScale: {e}",
            )


    def _prepare_payload_data(
        self,
        subscription: WOSSubscriptionInfoDict,
        response_obj: dict,
        start_time: datetime,
        end_time: datetime,
        kwargs,
    ) -> WOSPayloadLoggingObject:
        """
        Prepare the payload logging data for Watson OpenScale.
        """
        litellm_params = kwargs.get("litellm_params", {}) or {}
        metadata = litellm_params.get("metadata", {}) or {}
        # Get the prompt template subscription from metadata (if available)
        pl_dataset_id = (
            metadata.get(f"{self._metadata_suffix}_payload_logging_dataset_id") or subscription.get("payload_logging_dataset_id")
            or get_secret_str("WATSONX_GOV_PAYLOAD_LOGGING_DATASET_ID") or None
        )
        subscription_id = (
            metadata.get(f"{self._metadata_suffix}_subscription_id") or subscription.get("subscription_id")
            or get_secret_str("WATSONX_GOV_SUBSCRIPTION_ID") or None
        )
        deployment_id = (
            metadata.get(f"{self._metadata_suffix}_deployment_id") or subscription.get("deployment_id")
            or get_secret_str("WATSONX_GOV_DEPLOYMENT_ID") or None
        )
        infer_fields = metadata.get("infer_fields", None) or get_secret_bool("WATSONX_GOV_INFER_FIELDS") or True
        if pl_dataset_id is not None:
            pl_dataset = self.get_dataset_info(pl_dataset_id, infer_fields=infer_fields)
        elif subscription_id is not None:
            pl_dataset = self.get_dataset_info(type="payload_logging", target_type="subscription", target_id=str(subscription_id), infer_fields=infer_fields)
        elif deployment_id is not None:
            pl_dataset = self.get_dataset_info(type="payload_logging", target_type="deployment", target_id=str(deployment_id), infer_fields=infer_fields)
        else:
            raise ValueError(
                "No payload logging dataset found. Please provide a valid payload logging dataset id or subscription id."
            )
        # create the payload logging object
        return WOSPayloadLoggingObject(
            dataset_id=pl_dataset["id"],
            request={
                "fields": pl_dataset["input_fields"],
                "values": self._map_input_to_request(pl_dataset, kwargs)\
                            .get("values", []),
                "meta": {}
            },
            response={
                "fields": pl_dataset["output_fields"],
                "values": self._map_output_to_response(pl_dataset, response_obj, kwargs)\
                            .get("values", []),
                "meta": {}
            },
            response_time=(end_time - start_time).total_seconds()
        )

    def get_dataset_info(
        self,
        dataset_id: str | None = None,
        *,
        type: str | None = None,
        target_type: str | None = None,
        target_id: str | None = None,
        infer_fields: bool = False,
    ) -> WOSDatasetInfo:
        """
        Get the dataset info for the given dataset_id or type/target_type/target_id.
        If dataset_id is provided, it will return the dataset info for that dataset.
        If type/target_type/target_id is provided, it will return the dataset info for that type/target_type/target_id.
        """
        raise NotImplementedError(
            "get_dataset_info is not yet implemented"
        )

    @classmethod
    def _find_litellm_prompt(cls, project_id: str|None=None, space_id: str|None=None, **kwargs) -> "WatsonxGovPrompt":
        wx_params = _get_api_params(dict(**kwargs, project_id=project_id, space_id=space_id))
        if project_id is None and space_id is None:
            project_id = wx_params.get("project_id")
            space_id = wx_params.get("space_id")
        elif project_id is not None and space_id is not None:
            raise ValueError("Either one of project_id or space_id must be set")
        if (pt := wxgov_cache.get_cache(f"{project_id or space_id}-litellm-deployment")):
            return pt
        wx_mixin = IBMWatsonXMixin()
        base_url = wx_mixin._get_base_url(kwargs.get("api_base", None))
        headers = wx_mixin.validate_environment(
            headers={},
            model="",
            messages=[],
            optional_params={},
            api_base=base_url,
            litellm_params=kwargs.pop("litellm_params", {}),
            api_key=kwargs.get("api_key", None),
        )
        url = f"{base_url.rstrip('/')}/ml/v4/deployments"
        params = {k: v for k, v in wx_params.items() if v is not None}
        params['version'] = kwargs.get('version', litellm.WATSONX_DEFAULT_API_VERSION)
        verbose_logger.debug(f"[watsonx.governance] Fetching litellm deployment subscription with url: {url}")
        resp = litellm.module_level_client.get(url, headers=headers, params=params)
        deployment_resouces = sorted( # sort resources list for most recent
            resp.json().get("resources", []),
            key=lambda x: x.get("metadata", {}).get("created_at", ""), 
            reverse=True
        ) 
        verbose_logger.debug("[watsonx.governance] The litellm deployment subscription was fetched successfully")
        pt = None
        for r in deployment_resouces:
            if r.get("entity", {}).get("base_model_id") == "litellm":
                verbose_logger.debug(f"[watsonx.governance] Found litellm deployment subscription with id: {r.get('metadata', {}).get('id')}")
                litellm_deployment_info = r
                # get the prompt template from the deployment info
                prompt_id = litellm_deployment_info.get("entity", {}).get("prompt_template", {}).get("id")
                pt = WatsonxGovPrompt.get(prompt_id, project_id=project_id, space_id=space_id)
                if pt.subscription_id is None:
                    continue
                break
        if pt is None:
            raise ValueError(f"Litellm deployment subscription not found in {'space' if space_id else 'project'} {project_id or space_id}")
        wxgov_cache.set_cache(f"{project_id or space_id}-litellm-deployment", pt)
        return pt
    
    def _construct_input(self, kwargs):
        """Construct payload logging inputs with optional parameters"""
        inputs = {"messages": kwargs.get("messages")}
        if tools := kwargs.get("tools"):
            inputs["tools"] = tools

        for key in ["functions", "tools", "stream", "tool_choice", "user"]:
            if value := kwargs.get("optional_params", {}).pop(key, None):
                inputs[key] = value
        return inputs

    def _map_input_to_request(self, pl_dataset: WOSDatasetInfo, kwargs) -> dict:
        return {}

    def _map_output_to_response(self, pl_dataset: WOSDatasetInfo, response_obj, kwargs) -> dict:
        return {}

class WatsonxGovPrompt(BaseModel):
    """
    model for Watson OpenScale prompt template subscription details.
    """

    name: Optional[str] = None
    project_id: Optional[str] = None
    space_id: Optional[str] = None
    asset_id: Optional[str] = None
    subscription_id: Optional[str] = None
    payload_logging_dataset_id: Optional[str] = None
    deployment_id: Optional[str] = None
    mrm_monitor_instance_id: Optional[str] = None
    is_detached: bool = True

    @classmethod
    def get(cls, asset_id: str, **kwargs) -> "WatsonxGovPrompt":
        pt_info = _get_prompt_info(asset_id, **kwargs)
        subscription_details = _get_subscription_for_prompt(asset_id, **kwargs)
        return cls(
            name=pt_info.get("name"),
            project_id=kwargs.get("project_id"),
            space_id=kwargs.get("space_id"),
            asset_id=pt_info.get("id"),
            subscription_id=subscription_details.get("subscription_id"),
            payload_logging_dataset_id=subscription_details.get("payload_logging_dataset_id"),
            deployment_id=subscription_details.get("deployment_id"),
            mrm_monitor_instance_id=subscription_details.get("mrm_monitor_instance_id"),
        )


    def get_prompt_details(self) -> PromptAssetDetails:
        if self.asset_id is None:
            raise ValueError("Prompt id is missing. Please provide it to fetch prompt details.")
        prompt_info = _get_prompt_info(
            self.asset_id,
            project_id=self.project_id,
            space_id=self.space_id,
        )
        if self.name is None:
            self.name = prompt_info.get("name")
        return prompt_info
    
    def get_subscription_details(self) -> dict:
        if self.asset_id is None:
            raise ValueError("Prompt id is missing. Please provide it to fetch prompt details.")
        subs_info = _get_subscription_for_prompt(
            prompt_id=self.asset_id,
            project_id=self.project_id,
            space_id=self.space_id,
            deployment_id=self.deployment_id,
        )
        if self.subscription_id is None:
            self.subscription_id = subs_info.get("subscription_id")
        if self.payload_logging_dataset_id is None:
            self.payload_logging_dataset_id = subs_info.get("payload_logging_dataset_id")
        if self.deployment_id is None:
            self.deployment_id = subs_info.get("deployment_id")
        if self.mrm_monitor_instance_id is None:
            self.mrm_monitor_instance_id = subs_info.get("mrm_monitor_instance_id")
        return subs_info

def setup_detached_prompt_logging(
    prompt_id: Optional[str] = None,
    *,
    monitors: Optional[dict] = None,
    name: str = "LiteLLM Prompt Template",
    prompt: str | dict = DEFAULT_LITELLM_PROMPT_TEMPLATE,
    prompt_variables: list[str] | dict | None = None,
    model_id: str = "litellm",
    description: Optional[str] = "Created by LiteLLM's watsonx.governance Integration",
    detached: bool = True,
    **kwargs,
) -> WatsonxGovPrompt:
    """
    Set up a subscription in watsonx.governance for a detached prompt template valid for LiteLLM.

    Args:
        prompt_id (str, optional): The id of the prompt template to use. If not provided, a new prompt will be created.
        monitors (dict, optional): The monitors to set up for the prompt template. Defaults to a basic generative AI quality monitor.
        name (str, optional): The name of the prompt template. Defaults to "LiteLLM Prompt Template".
        prompt (str | dict, optional): The prompt template to use. Defaults to a basic chat template.
        prompt_variables (list[str] | dict, optional): The variables to use in the prompt template. Defaults to None.
        model_id (str, optional): The model id to use for the prompt template. Defaults to "litellm".
        description (str, optional): A description for the prompt template. Defaults to "Created by LiteLLM's watsonx.governance Integration".
        detached (bool, optional): Whether the prompt is detached or not. Defaults to True.
        
    Returns:
        WatsonxGovPrompt: The prompt template subscription details.
        
    Raises:
        ValueError: If the input prompt is not a string or a dictionary.
        TimeoutError: If the prompt setup takes too long to complete.
        RuntimeError: If the prompt setup fails.
    """
    api_params = _get_api_params(kwargs)
    # call the factsheets api to get the asset_id
    container_type = "project" if api_params.get("project_id") else "space"
    container_id = api_params.get("project_id") or api_params.get("space_id")
    if container_id is None:
        raise ValueError("Either project_id or space_id must be provided in the api_params.")
    if prompt_id is None:
        # create the prompt asset
        prompt_details = {}
        if isinstance(prompt, str):
            if prompt == DEFAULT_LITELLM_PROMPT_TEMPLATE and prompt_variables is None:
                prompt_details["prompt_variables"] = ["chat_history", "model"]
            elif prompt_variables is not None:
                prompt_details["prompt_variables"] = prompt_variables # type: ignore
            prompt_details["input"] = prompt # type: ignore
        elif isinstance(prompt, dict):
            prompt_details = prompt
        else:
            raise ValueError("input must be a string or a dictionary")
        prompt_info = _create_prompt_asset(
            name=name,
            model_id=model_id,
            prompt_details=prompt_details,
            container_type=container_type,
            container_id=container_id,
            description=description,
            detached=detached,
            **kwargs,
        )
        prompt_id = prompt_info["id"]
    else:
        prompt_info = _get_prompt_info(
            prompt_id, 
            project_id=api_params.get("project_id"),
            space_id=api_params.get("space_id"),
        )

    # create the prompt subscription in Watson OpenScale
    wx_mixin = IBMWatsonXMixin()
    base_url = wx_mixin._get_base_url(kwargs.pop("api_base", None))
    headers = wx_mixin.validate_environment(
        headers={},
        model="",
        messages=[],
        api_key=kwargs.pop("api_key", None),
        api_base=base_url,
        litellm_params=kwargs.pop("litellm_params", {}),
        optional_params=kwargs,
    )
    wos_url = _get_watson_openscale_url(
        base_url=base_url, 
        headers=headers, 
        **kwargs
    )
    if (get_secret_str("WATSONX_INSTANCE_ID") or "").lower() == "openshift":
        try:
            _create_wos_instance_mapping(
                wos_url=wos_url,
                headers=headers,
                container_id=container_id,
                container_type=container_type,
                service_instance_id=(
                    kwargs.get("openscale_instance_id") or 
                    get_secret_str("WATSONX_OPENSCALE_INSTANCE_ID") or
                    None
                ),
            )
        except:
            pass
    
    # create the prompt subscription
    setup_info = execute_prompt_setup(
        prompt_id=prompt_id,
        monitors=monitors,
        api_params=api_params,
        wos_url=wos_url,
        headers=headers,
        **kwargs,
    )

    return WatsonxGovPrompt(
        name=prompt_info.get("name"),
        asset_id=prompt_info.get("id"),
        subscription_id=setup_info.get("subscription_id"),
        payload_logging_dataset_id=setup_info.get("payload_logging_dataset_id"),
        deployment_id=setup_info.get("deployment_id"),
        mrm_monitor_instance_id=setup_info.get("mrm_monitor_instance_id"),
        project_id=api_params.get("project_id"),
        space_id=api_params.get("space_id"),
        is_detached=detached,
    )

def execute_prompt_setup(
    prompt_id: str,
    api_params: WatsonXAPIParams,
    wos_url: Optional[str] = None,
    *,
    headers: Optional[dict] = None,
    monitors: Optional[dict] = None,
    **kwargs,
): 
    wos_url = wos_url or _get_watson_openscale_url(
        base_url=api_params.get("api_base", None),
        headers={},
        **api_params,
    )
    if headers is None:
        wx_mixin = IBMWatsonXMixin()
        base_url = wx_mixin._get_base_url(kwargs.get("api_base", None))
        headers = wx_mixin.validate_environment(
            headers={},
            model="",
            messages=[],
            api_key=kwargs.get("api_key", None),
            api_base=base_url,
            litellm_params=kwargs.get("litellm_params", {}),
            optional_params=kwargs,
        )
    if monitors is None:
        monitors = {
            "generative_ai_quality": {
                "parameters": {"min_sample_size": 10, "metrics_configuration": {}}
            }
        }
    params = {
        "prompt_template_asset_id": prompt_id,
        "project_id": api_params.get("project_id"),
        "space_id": api_params.get("space_id"),
        "deployment_id": None,
    }
    params = {k: v for k, v in params.items() if v is not None}
    operational_id = (
        get_secret_str("WATSONX_OPENSCALE_OPERATIONAL_SPACE_ID") or "development"
    )
    payload = {
        "label_column": kwargs.pop("label_column", "reference"),
        "operational_space_id": kwargs.pop("operational_space_id", operational_id),
        "problem_type": kwargs.pop("problem_type", kwargs.pop("task_id", "generation")),
        "classification_type": kwargs.pop("classification_type", None),
        "input_data_type": kwargs.pop("input_data_type", "unstructured_text"),
        "context_fields": kwargs.pop("context_fields", None),
        "question_fields": kwargs.pop("question_fields", None),
        "monitors": monitors,
    }
    payload = {k:v for k, v in payload.items() if v is not None}
    url = f"{wos_url.rstrip('/')}/v2/prompt_setup"
    verbose_logger.debug(
        f"[watsonx.governance] Creating wos prompt subscription with url: {url} with payload:\n{payload}"
        f" and params:\n{params}"
    )
    sys_info = '{0} {1} {2}'.format(
        platform.system(),  # OS
        platform.release(),  # OS version
        platform.python_version()
    )
    headers.update({
         'Content-Type': 'application/json', 
         'Accept': 'application/json',
         'User-Agent': f'litellm-watsonx-governance-{sys_info}',
    })
    resp = litellm.module_level_client.post(url, headers=headers, json=payload, params=params, files=[])
    setup_info = resp.json()
    verbose_logger.info(
        f"[watsonx.governance] The prompt setup for prompt {setup_info.get('prompt_template_asset_id')} was initiated successfully.\n"
        "Waiting for the prompt setup to complete..."
    )
    tstart = time.time()
    while setup_info.get("subscription_id") is None and (setup_info.get("status") or {}).get("state", "") == "RUNNING":
        setup_info = _get_subscription_for_prompt(
            prompt_id=prompt_id,
            project_id=api_params.get("project_id"),
            space_id=api_params.get("space_id"),
            **kwargs,
        )
        if time.time() - tstart > 300:
            raise TimeoutError(f"Prompt setup took too long to complete. Last status: {setup_info}")
    verbose_logger.info(
        f"[watsonx.governance] The prompt setup for prompt {setup_info.get('prompt_template_asset_id')} was completed successfully."
    )
    return setup_info

def _create_prompt_asset(
    name: str,
    model_id: str,
    prompt_details: dict,
    container_type: str,
    container_id: str,
    description: str | None = None,
    detached: bool = True,
    **kwargs,
) -> PromptAssetDetails:
    wx_mixin = IBMWatsonXMixin()
    base_url = wx_mixin._get_base_url(kwargs.pop("api_base", None))
    headers = wx_mixin.validate_environment(
        headers={},
        model="",
        messages=[],
        api_key=kwargs.pop("api_key", None),
        api_base=base_url,
        litellm_params=kwargs.pop("litellm_params", {}),
        optional_params=kwargs,
    )
    body = {
        "name": name,
        "task_ids": [kwargs.pop("task_id", "generation")],
        "description": description or "",
        "prompt": {
            "model_id": model_id,
            "data": {},
        },
    }
    # update payload body based on prompt details
    prompt_body_vars = ["prompt_variables", "model_version"]
    body.update(
        {
            k: prompt_details.pop(k)
            for k in prompt_body_vars
            if k in prompt_details and prompt_details[k] is not None
        }
    )
    if body.get("prompt_variables"):
        prompt_variables = body["prompt_variables"]
        if isinstance(prompt_variables, dict):
            body["prompt_variables"] = {
                key: {"default_value": value} if not isinstance(value, dict) else value
                for key, value in prompt_variables.items()
            }
        elif isinstance(prompt_variables, list):
            body["prompt_variables"] = {key: {} for key in prompt_variables}
        else:
            raise ValueError("prompt_variables must be a list or a dictionary")
    prompt_vars = ["input", "model_parameters"]
    body["prompt"].update( # type: ignore
        {
            k: prompt_details.pop(k)
            for k in prompt_vars
            if k in prompt_details and prompt_details[k] is not None
        }
    )
    if "input" in body["prompt"]:
        body["prompt"]["input"] = [[body["prompt"]["input"], ""]] # type: ignore
    prompt_data_vars = [
        "instruction",
        "input_prefix",
        "output_prefix",
        "examples",
        "structured_examples",
    ]
    prompt_data = {
        k: prompt_details.pop(k) for k in prompt_data_vars if k in prompt_details
    }
    prompt_data = {k: v for k, v in prompt_data.items() if v is not None}
    if "structured_examples" in prompt_data:
        prompt_data["structured_examples"] = [
            [inp, out] for inp, out in prompt_data["structured_examples"].items()
        ]
    body["prompt"]["data"] = prompt_data # type: ignore
    if detached:
        # update body for detached prompt
        body["input_mode"] = "detached"
        body["prompt"]["external_information"] = { # type: ignore
            "external_prompt_id": kwargs.pop(
                "external_prompt_id", "litellm-detached-prompt"
            ),
            "external_model_id": kwargs.pop("external_model_id", model_id),
            "external_model_provider": kwargs.pop(
                "external_model_provider", "LiteLLM"
            ),
        }
        body["prompt"]["external_information"].update( # type: ignore
            {
                k: v
                for k, v in kwargs.items()
                if k.startswith("external_") and v is not None
            }
        )
    # make the call to create the detached prompt
    if (get_secret_str("WATSONX_INSTANCE_ID") or "").lower() != "openshift":
        base_url = "https://api.dataplatform.cloud.ibm.com"
    prompts_url = (
        f"{base_url.rstrip('/')}/wx/v1/prompts?{container_type}_id={container_id}"
    )
    verbose_logger.debug(
        f"Creating prompt subscription (detached: {detached}) with url: {prompts_url}"
    )
    resp = litellm.module_level_client.post(prompts_url, headers=headers, json=body)
    prompt_info = resp.json()
    verbose_logger.info(
        "The detached prompt with ID {} was created successfully in container_id {}.".format(
            prompt_info.get("id"), container_id
        )
    )
    return prompt_info

def _get_prompt_info(prompt_id: str, **kwargs) -> PromptAssetDetails:
    wx_mixin = IBMWatsonXMixin()
    base_url = wx_mixin._get_base_url(kwargs.pop("api_base", None))
    headers = wx_mixin.validate_environment(
        headers={},
        model="",
        messages=[],
        api_key=kwargs.pop("api_key", None),
        api_base=base_url,
        optional_params=kwargs,
        litellm_params={},
    )
    if (get_secret_str("WATSONX_INSTANCE_ID") or "").lower() != "openshift":
        base_url = "https://api.dataplatform.cloud.ibm.com"
    prompts_url = f"{base_url.rstrip('/')}/wx/v1/prompts/{prompt_id}"
    verbose_logger.debug(f"Fetching prompt info with url: {prompts_url}")
    params = dict(
        project_id=kwargs.get("project_id"),
        space_id=kwargs.get("space_id"),
    )
    params = {k: v for k, v in params.items() if v is not None}
    resp = litellm.module_level_client.get(prompts_url, headers=headers, params=params)
    prompt_info = resp.json()
    verbose_logger.info(f"The prompt with ID {prompt_id} was fetched successfully.")
    return prompt_info

def _get_subscription_for_prompt(
    prompt_id: str | None = None,
    deployment_id: str | None = None,
    project_id: str | None = None,
    space_id: str | None = None,
    **kwargs,
) -> dict:
    if project_id is None and space_id is None:
        raise ValueError("Either one of project_id or space_id must be set")
    elif project_id is not None and space_id is not None:
        raise ValueError("Only one of project_id or space_id must be set")
    headers = IBMWatsonXMixin().validate_environment(
        headers={},
        model="",
        messages=[],
        api_key=kwargs.get("api_key", None),
        api_base=kwargs.get("api_base", None),
        optional_params=kwargs,
        litellm_params={},
    )
    wos_url = kwargs.pop('wos_url', None) or _get_watson_openscale_url(
        base_url=kwargs.get('base_url'),
        headers=headers,
    )
    params = {
        "prompt_template_asset_id": prompt_id,
        "project_id": project_id,
        "space_id": space_id,
        "deployment_id": deployment_id,
    }
    params = {k: v for k, v in params.items() if v is not None}
    url = f"{wos_url}/v2/prompt_setup"
    verbose_logger.debug(f"[watsonx.governance] Fetching wos prompt setup status with url: {url}")
    resp = litellm.module_level_client.get(url, headers=headers, params=params)
    subscription_details = resp.json()
    verbose_logger.debug(f"[watsonx.governance] The prompt setup for prompt with id {prompt_id} was fetched successfully")
    return subscription_details

def _find_wos_service_instance_id(
    **kwargs,
) -> Optional[str]:
    # request the service instance id
    resources_url = "https://resource-controller.cloud.ibm.com/v2/resource_instances"
    if (headers := kwargs.pop("headers", None)) is None:
        headers = IBMWatsonXMixin().validate_environment(
            headers={},
            model="",
            messages=[],
            api_key=kwargs.pop("api_key", None),
            api_base=kwargs.pop("api_base", None),
            optional_params=kwargs,
            litellm_params={},
        )
    verbose_logger.debug(
        f"[watsonx.governance] Finding Watson OpenScale service instance with url: {resources_url}"
    )
    resources_response = litellm.module_level_client.get(
        resources_url, headers=headers
    )
    resources_json_resp = resources_response.json()
    resources = resources_json_resp['resources']
    next_url = resources_json_resp.get('next_url')
    instance_guid = None
    while instance_guid is None:
        for resource in resources:
            if resource["resource_id"] == "2ad019f3-0fd6-4c25-966d-f3952481a870":
                # this fixed id maps to Watson OpenScale's service on public cloud
                instance_guid = resource["guid"]
                break
        if instance_guid or next_url is None:
            break
        verbose_logger.debug(
            f"[watsonx.governance] Could not find. Fetching next page of resources with url: {next_url}"
        )
        resources_response = litellm.module_level_client.get(
            "https://resource-controller.cloud.ibm.com" + next_url, headers=headers)
        resources_json_resp = resources_response.json()
        resources = resources_json_resp['resources']
        next_url = resources_json_resp.get('next_url')
        
    if instance_guid is None:
        raise ValueError("Watson OpenScale service instance id not found")
    verbose_logger.info(
        f"[watsonx.governance] Found Watson OpenScale service instance with guid: {instance_guid}"
    )
    return instance_guid

def _create_wos_instance_mapping(
    wos_url: str,
    headers: dict,
    container_id: str,
    container_type: str,
    service_instance_id: Optional[str] = None,
) -> None:
    instance_mapping_url = f"{wos_url.rstrip('/')}/v2/instance_mappings"
    if service_instance_id is None:
        service_instance_id = "00000000-0000-0000-0000-000000000000"
    payload = {
        "service_instance_id": service_instance_id,
        "target": {
            "target_type": container_type,
            "target_id": container_id,
        },
    }
    verbose_logger.debug(
        f"[watsonx.governance] Creating instance mapping with url: {instance_mapping_url}"
    )
    _ = litellm.module_level_client.post(
        instance_mapping_url, headers=headers, json=payload
    )
    verbose_logger.debug(
        "[watsonx.governance] The openscale instance mapping was created successfully."
    )

def _get_watson_openscale_url(base_url=None, headers=None, **kwargs) -> str:
    wos_url_ = get_secret_str("WATSONX_OPENSCALE_API_BASE")
    if wos_url_:
        return wos_url_
    service_instance_id = kwargs.pop(
        "openscale_instance_id", get_secret_str("WATSONX_GOV_OPENSCALE_INSTANCE_ID")
    ) or None
    if base_url is None:
        wx_mixin = IBMWatsonXMixin()
        base_url = wx_mixin._get_base_url(kwargs.pop("api_base", None))
    if (get_secret_str("WATSONX_INSTANCE_ID") or "").lower() == "openshift":
        # non-SaaS watsonx instance
        if not service_instance_id:
            service_instance_id = "00000000-0000-0000-0000-000000000000" # default for non-SaaS watsonx
        wos_url = (base_url.rstrip("/") + f"/openscale/{service_instance_id}").rstrip("/")
    else:
        # SaaS watsonx instance
        if service_instance_id is None:
            cached_service_instance_id = wxgov_cache.get_cache("watsonx-gov-wos-service-instance-id")
            if cached_service_instance_id:
                service_instance_id = cached_service_instance_id
            else:
                verbose_logger.debug(
                    "Service instance id not found in cache. Attempting to find it."
                )
                service_instance_id = _find_wos_service_instance_id(base_url=base_url, headers=headers)
                wxgov_cache.set_cache("watsonx-gov-wos-service-instance-id", service_instance_id)
        wos_url = f"https://api.aiopenscale.cloud.ibm.com/openscale/{service_instance_id}"
    return wos_url