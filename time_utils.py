class TimeUtils:
    @staticmethod
    def convert_to_seconds(time_str):
        """Convert a time string in the format 'hours:minutes:seconds' to seconds."""
        hours, minutes, seconds = map(float, time_str.split(':'))
        total_seconds = hours * 3600 + minutes * 60 + seconds
        return total_seconds

    @staticmethod
    def convert_to_time_string(seconds):
        """Convert a number of seconds to a time string in the format 'hours:minutes:seconds'."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = seconds % 60
        return f"{hours}:{minutes}:{seconds:.2f}"

    @staticmethod
    def time_string_to_seconds(time_string):
        h, m, s = map(float, time_string.split(':'))
        return h * 3600 + m * 60 + s

    @staticmethod
    def extend_sample(start_time, end_time, total_length):
        """Extend a sample to be at least 1 minute long by adding time before and/or after it."""
        sample_length = end_time - start_time

        # If the sample is already at least 2 minutes long, return the original start and end times
        if sample_length >= 60:
            return start_time, end_time

        # Calculate how much time we need to add
        extra_time = 60 - sample_length

        # Split the extra time evenly between the start and end of the sample
        extra_start = extra_time / 2
        extra_end = extra_time / 2

        # Adjust the start and end times
        start_time -= extra_start
        end_time += extra_end

        # Make sure we don't go beyond the total length of the audio
        if start_time < 0:
            end_time += abs(start_time)
            start_time = 0
        if end_time > total_length:
            start_time -= (end_time - total_length)
            end_time = total_length

        return max(start_time, 0), min(end_time, total_length)
