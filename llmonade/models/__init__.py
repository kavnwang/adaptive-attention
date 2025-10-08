# NOTE: If additional model builders are added elsewhere in the project,
# import and expose them here so the registry stays centralized.
from llmonade.models.joyce_pretrain import build_model as build_joyce_pretrain

MODEL_BUILDERS = {
    # "transformer": build_transformer,
    "joyce_pretrain": build_joyce_pretrain,
}
