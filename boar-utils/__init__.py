from ._readers import AbstractReader


def get_reader_names() -> list:
    """Get reader names

    Returns:
        list: available readers
    """
    pass


def get_reader(name: str) -> AbstractReader:
    """Get reader by name

    Args:
        name (str): reader name. List of readers can be found by calling get_reader_names
    """
    pass
