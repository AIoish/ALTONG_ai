from datetime import datetime, timedelta

from pydantic import AliasChoices, BaseModel, Field, field_validator


def validate_utc_z_datetime(value: object) -> object:
    if isinstance(value, str):
        if not value.endswith("Z"):
            raise ValueError("datetime must use UTC ISO 8601 format ending in Z")
        return value

    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() != timedelta(0):
            raise ValueError("datetime must use UTC")

    return value


class RawNotification(BaseModel):
    id: str = Field(validation_alias=AliasChoices("id", "Id"))
    app_name: str = Field(validation_alias=AliasChoices("app_name", "appName", "AppName"))
    sender: str = Field(validation_alias=AliasChoices("sender", "Sender"))
    title: str = Field(validation_alias=AliasChoices("title", "Title"))
    body: str = Field(validation_alias=AliasChoices("body", "Body"))
    timestamp: datetime = Field(validation_alias=AliasChoices("timestamp", "Timestamp"))

    @field_validator("sender", mode="before")
    @classmethod
    def normalize_null_sender(cls, value: object) -> object:
        return "" if value is None else value
    _validate_timestamp = field_validator("timestamp", mode="before")(
        validate_utc_z_datetime
    )


MAX_RECENT_PROCESSES = 3


class CurrentContext(BaseModel):
    active_process: str = Field(
        validation_alias=AliasChoices("active_process", "activeProcess", "ActiveProcess")
    )
    window_title: str = Field(
        validation_alias=AliasChoices("window_title", "windowTitle", "WindowTitle")
    )
    last_updated: datetime = Field(
        validation_alias=AliasChoices("last_updated", "lastUpdated", "LastUpdated")
    )
    duration_seconds: int = Field(
        default=0,
        ge=0,
        strict=True,
        validation_alias=AliasChoices("duration_seconds", "durationSeconds", "DurationSeconds"),
    )
    recent_processes: list[str] = Field(
        default_factory=list,
        max_length=MAX_RECENT_PROCESSES,
        validation_alias=AliasChoices("recent_processes", "recentProcesses", "RecentProcesses"),
    )

    _validate_last_updated = field_validator("last_updated", mode="before")(
        validate_utc_z_datetime
    )

    @field_validator("recent_processes", mode="before")
    @classmethod
    def normalize_missing_recent_processes(cls, value: object) -> object:
        return [] if value is None else value

    @field_validator("recent_processes")
    @classmethod
    def validate_recent_process_names(cls, value: list[str]) -> list[str]:
        if any(not name.strip() for name in value):
            raise ValueError("recent_processes must contain non-empty process names")
        return value


class FilterLabel(BaseModel):
    urgency_score: int = Field(ge=1, le=5)
    relevance_score: int = Field(ge=1, le=5)
    category: str
    ai_summary_reason: str


class FilteringSample(BaseModel):
    notification: RawNotification
    context: CurrentContext
    label: FilterLabel
