"""Configuration schema using Pydantic."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings
from pydantic.alias_generators import to_camel


class Base(BaseModel):
    """Base model that accepts both camelCase and snake_case keys."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class AgentDefaults(Base):
    """Default agent configuration."""

    workspace: str = "~/.nanobot/workspace"
    model: str = "gemini/gemini-2.5-flash-preview"
    provider: str = "gemini"
    max_tokens: int = 8192
    temperature: float = 0.1
    max_tool_iterations: int = 40
    memory_window: int = 100
    reasoning_effort: str | None = None


class AgentsConfig(Base):
    """Agent configuration."""

    defaults: AgentDefaults = Field(default_factory=AgentDefaults)


class ProviderConfig(Base):
    """LLM provider configuration."""

    api_key: str = ""
    api_base: str | None = None
    extra_headers: dict[str, str] | None = None


class ProvidersConfig(Base):
    """Configuration for LLM providers."""

    gemini: ProviderConfig = Field(default_factory=ProviderConfig)


class WebSearchConfig(Base):
    """Web search tool configuration."""

    api_key: str = ""
    max_results: int = 5


class WebToolsConfig(Base):
    """Web tools configuration."""

    proxy: str | None = None
    search: WebSearchConfig = Field(default_factory=WebSearchConfig)


class ExecToolConfig(Base):
    """Shell exec tool configuration."""

    timeout: int = 60
    path_append: str = ""


class MCPServerConfig(Base):
    """MCP server connection configuration (stdio or HTTP)."""

    type: str | None = None
    command: str = ""
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    url: str = ""
    headers: dict[str, str] = Field(default_factory=dict)
    tool_timeout: int = 30


class ToolsConfig(Base):
    """Tools configuration."""

    web: WebToolsConfig = Field(default_factory=WebToolsConfig)
    exec: ExecToolConfig = Field(default_factory=ExecToolConfig)
    restrict_to_workspace: bool = False
    mcp_servers: dict[str, MCPServerConfig] = Field(default_factory=dict)


class Config(BaseSettings):
    """Root configuration for nanobot."""

    agents: AgentsConfig = Field(default_factory=AgentsConfig)
    providers: ProvidersConfig = Field(default_factory=ProvidersConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)

    @property
    def workspace_path(self) -> Path:
        """Get expanded workspace path."""
        return Path(self.agents.defaults.workspace).expanduser()

    def get_provider(self, model: str | None = None) -> ProviderConfig | None:
        """Get the Gemini provider config."""
        return self.providers.gemini if self.providers.gemini.api_key else None

    def get_provider_name(self, model: str | None = None) -> str | None:
        """Get the provider name."""
        return "gemini"

    def get_api_key(self, model: str | None = None) -> str | None:
        """Get API key."""
        return self.providers.gemini.api_key or None

    def get_api_base(self, model: str | None = None) -> str | None:
        """Get API base URL."""
        return self.providers.gemini.api_base

    model_config = ConfigDict(env_prefix="NANOBOT_", env_nested_delimiter="__")
