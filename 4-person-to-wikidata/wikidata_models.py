from typing import Dict, List, Optional

from pydantic import BaseModel, Field



class PersonRequest(BaseModel):
    person: str
    context: Optional[str] = None


class BirthdayResponse(BaseModel):
    person: str
    qid: str
    birthday: Optional[str] = None


class StudentsResponse(BaseModel):
    person: str
    qid: str
    students: List[Dict[str, str]]


class PoliticalPartyResponse(BaseModel):
    person: str
    qid: str
    political_party: List[Dict[str, str]]


class SupervisorResponse(BaseModel):
    person: str
    qid: str
    supervisors: List[Dict[str, str]]


class AllResponse(BaseModel):
    person: str
    qid: str
    birthday: Optional[str] = None
    students: List[Dict[str, str]]
