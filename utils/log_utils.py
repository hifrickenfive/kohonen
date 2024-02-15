def create_log(message, log_file_path):
    with open(log_file_path, "a") as log_file:
        log_file.write(message + "\n\n")
