from pipeline import Pipe
from time_utils import TimeUtils


class Ranker(Pipe):
    def __init__(self, name, top_k=10):
        super().__init__(name)
        self.top_k = top_k
        self.features = []

    def rank(self, features: dict):
        list_of_mentions_by_name = features.get('namedPeople', [])
        total_length = features['length']
        suspicious_time_stamps = {}
        for feature in self.features:
            if feature in features:
                feature_values = features[feature]
                for feature_value, mentions in feature_values.items():
                    for mention in mentions:
                        if mention['start'] not in suspicious_time_stamps:
                            suspicious_time_stamps[mention['start']] = []
                        suspicious_time_stamps[mention['start']].append((mention['end'], feature_value))

        # Sort the dictionary by the number of items in each list
        sorted_dict = dict(sorted(suspicious_time_stamps.items(), key=lambda item: len(item[1]), reverse=True))
        top_k = self.get_top_k(sorted_dict, list_of_mentions_by_name, total_length)
        return {"time_stamps": top_k, "transcript": features['transcript']}

    def get_top_k(self, suspicious_time_stamps, list_of_mentions_by_name, audio_length):
        """Get the top k suspicious time stamps."""
        top_k = []
        for key, value in suspicious_time_stamps.items():
            if len(top_k) == self.top_k:
                break
            description = ", ".join([tup[1] for tup in value]).strip(",")
            float_key = TimeUtils.convert_to_seconds(key)
            for mention in value:
                float_mention = TimeUtils.convert_to_seconds(mention[0])
                if len(top_k) == self.top_k:
                    break
                if float_mention - float_key > 5:  # 5 seconds
                    if not self.is_overlapping(float_key, float_mention, top_k):
                        top_k.append({'start': float_key, 'end': float_mention, 'description': description})
                    break
                else:
                    extended_start, extended_end = TimeUtils.extend_sample(float_key, float_mention, audio_length)
                    if not self.is_overlapping(extended_start, extended_end, top_k):
                        top_k.append({'start': extended_start, 'end': extended_end, 'description': description})
                    break
        # Sort the list by start time
        top_k = sorted(top_k, key=lambda k: k['start'])

        for item in top_k:
            start = TimeUtils.convert_to_time_string(item.get('start'))
            end = TimeUtils.convert_to_time_string(item.get('end'))
            item['start'] = start
            item['end'] = end
        return top_k

    @staticmethod
    def is_overlapping(start_time, end_time, time_spans):
        """Check if a given time span overlaps with any of the existing time spans."""
        for span in time_spans:
            if start_time < span['end'] and end_time > span['start']:
                return True
        return False

    def __call__(self, *args, **kwargs):
        return self.rank(*args, **kwargs)