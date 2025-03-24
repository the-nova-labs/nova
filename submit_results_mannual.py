from submit_results import submit_results
import json

def submit_results_mannual(epoch_number: int):
    with open(f"results/submissions_epoch_{epoch_number}.json", "r") as f:
        commitments = json.load(f)        
    try:
        submit_results(commitments)
    except Exception as e:
        print(e)


if __name__ == "__main__":
    submit_results_mannual(14430)