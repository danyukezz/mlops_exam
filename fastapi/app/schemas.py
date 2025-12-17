from pydantic import BaseModel

class PredictRequest(BaseModel):
    region: str
    primary_role: str
    alignment: str
    status: str
    species: str
    honour_1to5: int
    ruthlessness_1to5: int
    intelligence_1to5: int
    combat_skill_1to5: int
    diplomacy_1to5: int
    leadership_1to5: int
    trait_loyal: bool
    trait_scheming: bool

class PredictResponse(BaseModel):
    house: str
