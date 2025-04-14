import typer
from ray import tune

app = typer.Typer()


@app.command()
def main(
        dataset_root_dir: str = typer.Argument(..., help="Path to the dataset root directory."),
        search_results_output_dir: str = typer.Argument(..., help="Path to save search results."),
):
    """
    Main function to perform search and save results.

    Args:
        dataset_root_dir (str): Path to the dataset root directory.
        search_results_output_dir (str): Path to save search results.
    """
    search_space = {
        "self_attention_heads": tune.choice([1, 2, 4, 8, 16, 32]),
        "embedding_dim": tune.choice([32, 64, 128, 256, 512, 728, 1024]),
        "self_attn_ff_dim": tune.choice([64, 128, 256, 512, 728, 1024]),
        "num_cross_modal_attention_blocks": tune.choice([1, 2, 4, 8]),
        "num_cross_modal_attention_heads": tune.choice([1, 2, 4, 8, 16, 32]),
    }
