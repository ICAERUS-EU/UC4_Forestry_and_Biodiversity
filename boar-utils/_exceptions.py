

class SrtValidationError(Exception):
    def __init__(self, message=None):            
        # Call the base class constructor with the parameters it needs
        if message is None:
            self.message = "SRT parsing error, file type auto detection failed"
        else:
            self.message = message
        super().__init__(message)
        