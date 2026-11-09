"""Experiment-only formatter: insert fields once; preserve literal braces in data."""
from ais_bench.benchmark.openicl.icl_prompt_template import PromptTemplate
from ais_bench.benchmark.utils.prompt import PromptList

def _render_once(value, entry):
    if isinstance(value, str):
        return value.format(**entry)
    if isinstance(value, PromptList):
        return PromptList([_render_once(item, entry) for item in value])
    if isinstance(value, list):
        return [_render_once(item, entry) for item in value]
    if isinstance(value, dict):
        return {key: _render_once(item, entry) for key, item in value.items()}
    return value

class LiteralLongBenchPrompt(PromptTemplate):
    """Only the zero-shot LongBench template used by this experiment."""
    def generate_item(self, entry, output_field=None,
                      output_field_replace_token="", ice_field_replace_token=""):
        if self.ice_token is not None or self.sep_token is not None:
            raise ValueError("This experiment uses a zero-shot template without ICE/separator tokens.")
        if self.prompt_type != "meta":
            raise ValueError("Expected LongBench meta prompt template.")
        fields = dict(entry)
        if output_field is not None:
            fields[output_field] = output_field_replace_token
        encoded = self._encode_template(self.template, ice=False)
        return _render_once(encoded, fields)
