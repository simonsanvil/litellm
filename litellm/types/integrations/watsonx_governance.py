from typing import Optional, TypedDict, Any


class ScoringBody(TypedDict):
    """
    Watsonx Payload Logging Request
    """
    fields: list[str]
    values: list[list[str | int | float | bool]]
    meta: Optional[dict]

class WOSPayloadLoggingObject(TypedDict):
    """
    Represents a Watson Openscale Payload Logging Object
    """
    dataset_id: str
    request: ScoringBody
    response: ScoringBody
    response_time: float

class WOSDatasetInfo(TypedDict):
    """
    Represents a Watsonx OpenScale Dataset Info Object
    """
    id: str
    type: str
    input_fields: list[str]
    output_fields: list[str]
    
    # def map_output_fields(self, other_output_fields: list[str]) -> list[Any]:
    #     """
    #     Maps the output fields to the given other_output_fields.
    #     """
    #     return 

class WOSSubscriptionInfoDict(TypedDict):
    """
    Represents a Watsonx OpenScale Subscription Info Object
    """
    id: str
    name: str
    description: Optional[str]
    model_id: str
    asset_id: str
    asset_type: str
    payload_logging_dataset_id: Optional[str]
    feedback_dataset_id: Optional[str]
    project_id: Optional[str]
    space_id: Optional[str]
    input_fields: list[str]
    output_fields: list[str]
    credentials: dict[str, Any]

class PromptAssetDetails(TypedDict):
    """
    Represents the details of a prompt asset in Watsonx OpenScale.
    """
    id: str
    name: str
    description: Optional[str]
    model_id: str
    metadata: Optional[dict]
    prompt_variables: Optional[list[str]]
    project_id: Optional[str]
    space_id: Optional[str]
