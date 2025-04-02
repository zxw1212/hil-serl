from experiments.ram_insertion.config import TrainConfig as RAMInsertionTrainConfig
from experiments.usb_pickup_insertion.config import TrainConfig as USBPickupInsertionTrainConfig
from experiments.banana_pick_place.config import TrainConfig as BananaPickPlaceTrainConfig
from experiments.object_handover.config import TrainConfig as ObjectHandoverTrainConfig
from experiments.egg_flip.config import TrainConfig as EggFlipTrainConfig

CONFIG_MAPPING = {
                "ram_insertion": RAMInsertionTrainConfig,
                "usb_pickup_insertion": USBPickupInsertionTrainConfig,
                "banana_pick_place": BananaPickPlaceTrainConfig,
                "object_handover": ObjectHandoverTrainConfig,
                "egg_flip": EggFlipTrainConfig,
               }