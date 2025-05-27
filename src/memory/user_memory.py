class UserMemory:
    def __init__(self):
        self.memory = {}

    def get_summary(self, user_id):
        return "\n".join(self.memory.get(user_id, []))

    def update(self, user_id, user_msg, bot_msg):
        if user_id not in self.memory:
            self.memory[user_id] = []
        self.memory[user_id].append(f"User: {user_msg}")
        self.memory[user_id].append(f"Bot: {bot_msg}")
        self.memory[user_id] = self.memory[user_id][-10:]
