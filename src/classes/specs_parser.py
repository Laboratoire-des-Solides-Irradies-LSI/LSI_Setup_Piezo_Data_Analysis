import configparser
import os

class SpecsParser:
    def __init__(self, file) -> None:
        """
        Initialize the Config object.
        
        Args:
            file (str): The path to the configuration file.
        """
        self.file = file
        self.config = configparser.ConfigParser()
        if not os.path.isfile(self.file):
            raise FileNotFoundError(f"File '{self.file}' does not exist.")
        self.read()

    def read(self) -> None:
        """
        Read the configuration file. The config object now contains the variables and values from the file."""
        self.config.read(self.file)

    def get_section(self, section_name) -> dict:
        """
        Retrieve a specific section from the configuration as a dictionary.
        
        Args:
            section_name (str): The name of the section to retrieve.
        
        Returns:
            dict: A dictionary containing the key-value pairs for each variable of the section,
                  or an empty dictionary if the section does not exist.
        """
        # Check if the section exists in the config file and return it
        if section_name in self.config:
            return self.config[section_name]
        # Otherwise, return an empty dictionnary
        return dict()