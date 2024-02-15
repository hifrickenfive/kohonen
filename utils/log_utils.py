def create_log_message(log):
    log_message = (
        f"Datetime: {log['Datetime']}\n"
        f"Config: {log['Config']} \n"
        f"Elapsed time: {log['Elapsed time']:.3f} seconds\n"
        f"Final score: {log['final_av_dist_to_bmu']:.3f}. final_av_dist_to_bmu. Lower is better"
    )
    return log_message


def append_to_log_file(log_message, log_file_path):
    with open(log_file_path, "a") as log_file:
        log_file.write(log_message + "\n\n")
