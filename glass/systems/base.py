import datetime
import subprocess
import time
from abc import ABC, abstractmethod
from typing import List, Literal, Optional

from pydantic import BaseModel

from glass.config.schema import SystemConfig
from glass.datasets.base import EvaluationSample


class RawOutput(BaseModel):
    sample_id: str
    system_name: str
    model: Optional[str] = None  # Model used for inference (logged for reproducibility)
    command: List[str]  # AP-8: exact command invoked (required for reproducibility)
    prompt: str  # Full prompt sent via stdin for debugging/reproducibility
    output: str
    latency_s: float
    exit_code: int
    stderr: str
    error_type: Optional[Literal["timeout", "api_error", "refusal", "malformed", "crash"]] = None
    timestamp: str


class SystemUnderTest(ABC):
    def __init__(self, config: SystemConfig):
        self.config = config

    @abstractmethod
    def generate(self, sample: EvaluationSample) -> RawOutput:
        """Invoke the system on the sample and return the raw output."""
        pass

    def _run_command(self, command: List[str], prompt: str, sample_id: str) -> RawOutput:
        """Helper to run a subprocess with the given command and prompt."""
        start_time = time.time()
        timestamp = datetime.datetime.now().isoformat()
        model = self.config.model

        try:
            # AP-5: Use list of arguments (command)
            # AP-27: Handle encoding
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            stdout_bytes, stderr_bytes = process.communicate(
                input=prompt.encode("utf-8"), timeout=self.config.timeout_s
            )

            stdout = stdout_bytes.decode("utf-8", errors="replace")
            stderr = stderr_bytes.decode("utf-8", errors="replace")
            exit_code = process.returncode
            latency = time.time() - start_time

            error_type = None
            if exit_code != 0:
                error_type = "crash"

            return RawOutput(
                sample_id=sample_id,
                system_name=self.config.name,
                model=model,
                command=command,
                prompt=prompt,
                output=stdout,
                latency_s=latency,
                exit_code=exit_code,
                stderr=stderr,
                error_type=error_type,
                timestamp=timestamp,
            )

        except subprocess.TimeoutExpired:
            process.kill()
            latency = time.time() - start_time
            return RawOutput(
                sample_id=sample_id,
                system_name=self.config.name,
                model=model,
                command=command,
                prompt=prompt,
                output="",
                latency_s=latency,
                exit_code=124,  # Timeout
                stderr="TimeoutExpired",
                error_type="timeout",
                timestamp=timestamp,
            )
        except Exception as e:
            latency = time.time() - start_time
            return RawOutput(
                sample_id=sample_id,
                system_name=self.config.name,
                model=model,
                command=command,
                prompt=prompt,
                output="",
                latency_s=latency,
                exit_code=1,
                stderr=f"System Execution Error: {str(e)}",
                error_type="crash",
                timestamp=timestamp,
            )
