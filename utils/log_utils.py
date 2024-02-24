def create_log_message(log):
    log_message = (
        f"fatetime: {log['datetime']}\n"
        f"Config: {log['config']} \n"
        f"Elapsed time: {log['elapsed_time']:.3f} seconds\n"
        f"Final score: {log['av_gradient_magnitude']:.3f}. Average gradient magnitude of final image. Lower is better"
    )
    return log_message


def append_to_log_file(log_message, log_file_path):
    with open(log_file_path, "a") as log_file:
        log_file.write(log_message + "\n\n")
