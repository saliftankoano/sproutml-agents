# Temporary mock for agents module until the real one is available
# This allows the application to start without the agents dependency

class Agent:
    def __init__(self, name=None, instructions=None, handoff_description=None, handoffs=None):
        self.name = name
        self.instructions = instructions
        self.handoff_description = handoff_description
        self.handoffs = handoffs or []
    
    def __repr__(self):
        return f"Agent(name='{self.name}')"

class InputGuardrail:
    pass

class GuardrailFunctionOutput:
    pass

class MockResult:
    def __init__(self, output):
        self.final_output = output

class Runner:
    @staticmethod
    async def run(agent, prompt):
        # Mock implementation - replace with real agent logic
        return MockResult(f"Mock response from {agent.name}: Processed request '{prompt[:100]}...'")

class InputGuardrailTripwireTriggered(Exception):
    pass
