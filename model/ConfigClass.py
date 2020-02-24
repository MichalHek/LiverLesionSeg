import json

class ConfigClass():
    __instance = None
    @staticmethod
    def getInstance():
        """ Static access method. """
        if ConfigClass.__instance == None:
            ConfigClass()
        return ConfigClass.__instance
    def __init__(self, config_path=None):
        if ConfigClass.__instance != None or False:
            raise Exception("This class is a singleton!")
        else:
            if config_path:
                if config_path is None:
                    raise ValueError('You must supply model config path')
                with open(config_path) as json_file:
                    config_dict = json.load(json_file)

                for key in config_dict:
                    setattr(self, key, config_dict[key])

                if ('img_width' in config_dict) and ('img_height' in config_dict) and ('img_z' in config_dict):
                    self.input_shape = (self.img_width, self.img_height, self.img_z)

            ConfigClass.__instance = self

    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")

    def write_to_file(self, filepath=None, model=None, **kwargs):
        """"
        Write model config and architecture layers into json for tracability
        """
        import os
        if filepath is None:
            raise ValueError("Please provide file path for log info file (logdir+'log_infot.json')")
        base_path = os.path.split(filepath)[0]
        if not os.path.exists(base_path):
            os.makedirs(base_path)
        config_dict = self.__dict__
        if model:
            config_dict['model_layers'] = []
            for i,l in enumerate(model.layers):
                config_dict['model_layers'].append(str(l.output))

        for key, value in kwargs.items():
            config_dict[key] = value

        # Save dict as json
        with open(filepath, 'w') as outfile:
            json.dump(config_dict, outfile, indent=4, sort_keys=True)



