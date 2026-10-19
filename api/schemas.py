"""Request / response models for the prediction API."""

from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class PatientData(BaseModel):
    """One patient record. Field ranges follow the UCI documentation."""

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "age": 58, "sex": 1, "cp": 4, "trestbps": 140, "chol": 260,
            "fbs": 0, "restecg": 2, "thalach": 130, "exang": 1,
            "oldpeak": 2.4, "slope": 2, "ca": 2, "thal": 7,
        }
    })

    age: float = Field(..., ge=1, le=120, description="age in years")
    sex: Literal[0, 1] = Field(..., description="1 = male, 0 = female")
    cp: Literal[1, 2, 3, 4] = Field(
        ..., description="chest pain type: 1 typical angina, 2 atypical, "
                         "3 non-anginal, 4 asymptomatic")
    trestbps: float = Field(..., ge=50, le=250,
                            description="resting blood pressure (mm Hg)")
    chol: float = Field(..., ge=50, le=700,
                        description="serum cholesterol (mg/dl)")
    fbs: Literal[0, 1] = Field(..., description="fasting blood sugar > 120")
    restecg: Literal[0, 1, 2] = Field(..., description="resting ECG result")
    thalach: float = Field(..., ge=50, le=250,
                           description="max heart rate achieved")
    exang: Literal[0, 1] = Field(..., description="exercise induced angina")
    oldpeak: float = Field(..., ge=0, le=10,
                           description="ST depression induced by exercise")
    slope: Literal[1, 2, 3] = Field(..., description="slope of peak ST segment")
    ca: Optional[int] = Field(None, ge=0, le=3,
                              description="major vessels coloured (0-3), optional")
    thal: Optional[Literal[3, 6, 7]] = Field(
        None, description="3 normal, 6 fixed defect, 7 reversible defect, optional")


class PredictionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    prediction: int = Field(..., description="1 = heart disease, 0 = no disease")
    label: str
    confidence: float = Field(..., description="probability of the predicted class")
    probability_disease: float
    risk_level: str
    model_version: str


class BatchRequest(BaseModel):
    patients: List[PatientData] = Field(..., min_length=1, max_length=100)


class BatchResponse(BaseModel):
    predictions: List[PredictionResponse]
