class User:
    def __init__(self, name, startMood, endMood, favoriteCluster):
        self.name = name
        self.startMood = startMood
        self.endMood = endMood
        self.preferredCluster = favoriteCluster
    def get_profile(self):
        return {
            "name": self.name,
            "startMood": self.startMood,
            "endMood": self.endMood,
            "preferredCluster": self.preferredCluster
        }
