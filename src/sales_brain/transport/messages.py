class ModelPrompt(str):
    """String-compatible internal request with trusted instructions separate from input data."""
    def __new__(cls, system, user):
        value = super().__new__(cls, system + "\n" + user)
        value.system_message = system
        value.user_message = user
        return value
