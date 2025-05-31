from pydantic import BaseModel

class new_customer(BaseModel):
    EXT_SOURCE_3: float
    EXT_SOURCE_2: float
    EXT_SOURCE_1: float
    AMT_GOODS_PRICE: float
    AMT_ANNUITY: float
    FLAG_OWN_CAR: str
    NAME_EDUCATION_TYPE: str
    AMT_CREDIT: float
    DAYS_EMPLOYED: float
    DAYS_BIRTH: float