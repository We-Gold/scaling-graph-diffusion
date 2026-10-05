"""Minimal step runner. Steps share one `context` dict (model, datasets, device, alphas, ...)."""


class PipelineStep:
    def __init__(self, name):
        self.name = name

    def execute(self, context):
        raise NotImplementedError


class PipelineRunner:
    def __init__(self, steps):
        self.steps = steps

    def run(self, context):
        for step in self.steps:
            print(f"--- {step.name} ---")
            step.execute(context)
        return context
