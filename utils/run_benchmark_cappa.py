import argparse
import os
from functools import partial
from subprocess import call

import cv2
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax
import torch
import torchvision
import wandb
from flax.training import checkpoints, orbax_utils
from jax.experimental.mesh_utils import create_device_mesh
from jax.experimental.pjit import pjit
from jax.lax import with_sharding_constraint
from jax.sharding import Mesh, PartitionSpec
from torchvision.datasets import ImageNet
from tqdm import tqdm

from clip_jax import CLIPModel
from clip_jax.data import shift_tokens_left
from clip_jax.tokenizer import AutoTokenizer
from clip_jax.utils import load_config

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--tokenizer_name", type=str, default="openai/clip-vit-base-patch32")
parser.add_argument("--train_run", required=True, type=str, help='wandb run id as "entity/project/run_id"')
parser.add_argument("--latest_only", action="store_true", help="Evaluate all checkpoints")
parser.add_argument("--fraction", type=float, default=1.0, help="Fraction of dataset to use")
args = parser.parse_args()


def load_dataset(folder="benchmark_data", transform=None, **kwargs):
    """Load ImageNet"""

    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
        call(
            f"wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz --output-document={folder}/ILSVRC2012_devkit_t12.tar.gz",
            shell=True,
        )
        call(
            f"wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar --output-document={folder}/ILSVRC2012_img_val.tar",
            shell=True,
        )

    ds = ImageNet(root=folder, split="val", transform=transform, **kwargs)
    # use classnames from OpenAI
    ds.classes = classnames
    return ds


classnames = [
    "tench",
    "goldfish",
    "great white shark",
    "tiger shark",
    "hammerhead shark",
    "electric ray",
    "stingray",
    "rooster",
    "hen",
    "ostrich",
    "brambling",
    "goldfinch",
    "house finch",
    "junco",
    "indigo bunting",
    "American robin",
    "bulbul",
    "jay",
    "magpie",
    "chickadee",
    "American dipper",
    "kite (bird of prey)",
    "bald eagle",
    "vulture",
    "great grey owl",
    "fire salamander",
    "smooth newt",
    "newt",
    "spotted salamander",
    "axolotl",
    "American bullfrog",
    "tree frog",
    "tailed frog",
    "loggerhead sea turtle",
    "leatherback sea turtle",
    "mud turtle",
    "terrapin",
    "box turtle",
    "banded gecko",
    "green iguana",
    "Carolina anole",
    "desert grassland whiptail lizard",
    "agama",
    "frilled-necked lizard",
    "alligator lizard",
    "Gila monster",
    "European green lizard",
    "chameleon",
    "Komodo dragon",
    "Nile crocodile",
    "American alligator",
    "triceratops",
    "worm snake",
    "ring-necked snake",
    "eastern hog-nosed snake",
    "smooth green snake",
    "kingsnake",
    "garter snake",
    "water snake",
    "vine snake",
    "night snake",
    "boa constrictor",
    "African rock python",
    "Indian cobra",
    "green mamba",
    "sea snake",
    "Saharan horned viper",
    "eastern diamondback rattlesnake",
    "sidewinder rattlesnake",
    "trilobite",
    "harvestman",
    "scorpion",
    "yellow garden spider",
    "barn spider",
    "European garden spider",
    "southern black widow",
    "tarantula",
    "wolf spider",
    "tick",
    "centipede",
    "black grouse",
    "ptarmigan",
    "ruffed grouse",
    "prairie grouse",
    "peafowl",
    "quail",
    "partridge",
    "african grey parrot",
    "macaw",
    "sulphur-crested cockatoo",
    "lorikeet",
    "coucal",
    "bee eater",
    "hornbill",
    "hummingbird",
    "jacamar",
    "toucan",
    "duck",
    "red-breasted merganser",
    "goose",
    "black swan",
    "tusker",
    "echidna",
    "platypus",
    "wallaby",
    "koala",
    "wombat",
    "jellyfish",
    "sea anemone",
    "brain coral",
    "flatworm",
    "nematode",
    "conch",
    "snail",
    "slug",
    "sea slug",
    "chiton",
    "chambered nautilus",
    "Dungeness crab",
    "rock crab",
    "fiddler crab",
    "red king crab",
    "American lobster",
    "spiny lobster",
    "crayfish",
    "hermit crab",
    "isopod",
    "white stork",
    "black stork",
    "spoonbill",
    "flamingo",
    "little blue heron",
    "great egret",
    "bittern bird",
    "crane bird",
    "limpkin",
    "common gallinule",
    "American coot",
    "bustard",
    "ruddy turnstone",
    "dunlin",
    "common redshank",
    "dowitcher",
    "oystercatcher",
    "pelican",
    "king penguin",
    "albatross",
    "grey whale",
    "killer whale",
    "dugong",
    "sea lion",
    "Chihuahua",
    "Japanese Chin",
    "Maltese",
    "Pekingese",
    "Shih Tzu",
    "King Charles Spaniel",
    "Papillon",
    "toy terrier",
    "Rhodesian Ridgeback",
    "Afghan Hound",
    "Basset Hound",
    "Beagle",
    "Bloodhound",
    "Bluetick Coonhound",
    "Black and Tan Coonhound",
    "Treeing Walker Coonhound",
    "English foxhound",
    "Redbone Coonhound",
    "borzoi",
    "Irish Wolfhound",
    "Italian Greyhound",
    "Whippet",
    "Ibizan Hound",
    "Norwegian Elkhound",
    "Otterhound",
    "Saluki",
    "Scottish Deerhound",
    "Weimaraner",
    "Staffordshire Bull Terrier",
    "American Staffordshire Terrier",
    "Bedlington Terrier",
    "Border Terrier",
    "Kerry Blue Terrier",
    "Irish Terrier",
    "Norfolk Terrier",
    "Norwich Terrier",
    "Yorkshire Terrier",
    "Wire Fox Terrier",
    "Lakeland Terrier",
    "Sealyham Terrier",
    "Airedale Terrier",
    "Cairn Terrier",
    "Australian Terrier",
    "Dandie Dinmont Terrier",
    "Boston Terrier",
    "Miniature Schnauzer",
    "Giant Schnauzer",
    "Standard Schnauzer",
    "Scottish Terrier",
    "Tibetan Terrier",
    "Australian Silky Terrier",
    "Soft-coated Wheaten Terrier",
    "West Highland White Terrier",
    "Lhasa Apso",
    "Flat-Coated Retriever",
    "Curly-coated Retriever",
    "Golden Retriever",
    "Labrador Retriever",
    "Chesapeake Bay Retriever",
    "German Shorthaired Pointer",
    "Vizsla",
    "English Setter",
    "Irish Setter",
    "Gordon Setter",
    "Brittany dog",
    "Clumber Spaniel",
    "English Springer Spaniel",
    "Welsh Springer Spaniel",
    "Cocker Spaniel",
    "Sussex Spaniel",
    "Irish Water Spaniel",
    "Kuvasz",
    "Schipperke",
    "Groenendael dog",
    "Malinois",
    "Briard",
    "Australian Kelpie",
    "Komondor",
    "Old English Sheepdog",
    "Shetland Sheepdog",
    "collie",
    "Border Collie",
    "Bouvier des Flandres dog",
    "Rottweiler",
    "German Shepherd Dog",
    "Dobermann",
    "Miniature Pinscher",
    "Greater Swiss Mountain Dog",
    "Bernese Mountain Dog",
    "Appenzeller Sennenhund",
    "Entlebucher Sennenhund",
    "Boxer",
    "Bullmastiff",
    "Tibetan Mastiff",
    "French Bulldog",
    "Great Dane",
    "St. Bernard",
    "husky",
    "Alaskan Malamute",
    "Siberian Husky",
    "Dalmatian",
    "Affenpinscher",
    "Basenji",
    "pug",
    "Leonberger",
    "Newfoundland dog",
    "Great Pyrenees dog",
    "Samoyed",
    "Pomeranian",
    "Chow Chow",
    "Keeshond",
    "brussels griffon",
    "Pembroke Welsh Corgi",
    "Cardigan Welsh Corgi",
    "Toy Poodle",
    "Miniature Poodle",
    "Standard Poodle",
    "Mexican hairless dog (xoloitzcuintli)",
    "grey wolf",
    "Alaskan tundra wolf",
    "red wolf or maned wolf",
    "coyote",
    "dingo",
    "dhole",
    "African wild dog",
    "hyena",
    "red fox",
    "kit fox",
    "Arctic fox",
    "grey fox",
    "tabby cat",
    "tiger cat",
    "Persian cat",
    "Siamese cat",
    "Egyptian Mau",
    "cougar",
    "lynx",
    "leopard",
    "snow leopard",
    "jaguar",
    "lion",
    "tiger",
    "cheetah",
    "brown bear",
    "American black bear",
    "polar bear",
    "sloth bear",
    "mongoose",
    "meerkat",
    "tiger beetle",
    "ladybug",
    "ground beetle",
    "longhorn beetle",
    "leaf beetle",
    "dung beetle",
    "rhinoceros beetle",
    "weevil",
    "fly",
    "bee",
    "ant",
    "grasshopper",
    "cricket insect",
    "stick insect",
    "cockroach",
    "praying mantis",
    "cicada",
    "leafhopper",
    "lacewing",
    "dragonfly",
    "damselfly",
    "red admiral butterfly",
    "ringlet butterfly",
    "monarch butterfly",
    "small white butterfly",
    "sulphur butterfly",
    "gossamer-winged butterfly",
    "starfish",
    "sea urchin",
    "sea cucumber",
    "cottontail rabbit",
    "hare",
    "Angora rabbit",
    "hamster",
    "porcupine",
    "fox squirrel",
    "marmot",
    "beaver",
    "guinea pig",
    "common sorrel horse",
    "zebra",
    "pig",
    "wild boar",
    "warthog",
    "hippopotamus",
    "ox",
    "water buffalo",
    "bison",
    "ram (adult male sheep)",
    "bighorn sheep",
    "Alpine ibex",
    "hartebeest",
    "impala (antelope)",
    "gazelle",
    "arabian camel",
    "llama",
    "weasel",
    "mink",
    "European polecat",
    "black-footed ferret",
    "otter",
    "skunk",
    "badger",
    "armadillo",
    "three-toed sloth",
    "orangutan",
    "gorilla",
    "chimpanzee",
    "gibbon",
    "siamang",
    "guenon",
    "patas monkey",
    "baboon",
    "macaque",
    "langur",
    "black-and-white colobus",
    "proboscis monkey",
    "marmoset",
    "white-headed capuchin",
    "howler monkey",
    "titi monkey",
    "Geoffroy's spider monkey",
    "common squirrel monkey",
    "ring-tailed lemur",
    "indri",
    "Asian elephant",
    "African bush elephant",
    "red panda",
    "giant panda",
    "snoek fish",
    "eel",
    "silver salmon",
    "rock beauty fish",
    "clownfish",
    "sturgeon",
    "gar fish",
    "lionfish",
    "pufferfish",
    "abacus",
    "abaya",
    "academic gown",
    "accordion",
    "acoustic guitar",
    "aircraft carrier",
    "airliner",
    "airship",
    "altar",
    "ambulance",
    "amphibious vehicle",
    "analog clock",
    "apiary",
    "apron",
    "trash can",
    "assault rifle",
    "backpack",
    "bakery",
    "balance beam",
    "balloon",
    "ballpoint pen",
    "Band-Aid",
    "banjo",
    "baluster / handrail",
    "barbell",
    "barber chair",
    "barbershop",
    "barn",
    "barometer",
    "barrel",
    "wheelbarrow",
    "baseball",
    "basketball",
    "bassinet",
    "bassoon",
    "swimming cap",
    "bath towel",
    "bathtub",
    "station wagon",
    "lighthouse",
    "beaker",
    "military hat (bearskin or shako)",
    "beer bottle",
    "beer glass",
    "bell tower",
    "baby bib",
    "tandem bicycle",
    "bikini",
    "ring binder",
    "binoculars",
    "birdhouse",
    "boathouse",
    "bobsleigh",
    "bolo tie",
    "poke bonnet",
    "bookcase",
    "bookstore",
    "bottle cap",
    "hunting bow",
    "bow tie",
    "brass memorial plaque",
    "bra",
    "breakwater",
    "breastplate",
    "broom",
    "bucket",
    "buckle",
    "bulletproof vest",
    "high-speed train",
    "butcher shop",
    "taxicab",
    "cauldron",
    "candle",
    "cannon",
    "canoe",
    "can opener",
    "cardigan",
    "car mirror",
    "carousel",
    "tool kit",
    "cardboard box / carton",
    "car wheel",
    "automated teller machine",
    "cassette",
    "cassette player",
    "castle",
    "catamaran",
    "CD player",
    "cello",
    "mobile phone",
    "chain",
    "chain-link fence",
    "chain mail",
    "chainsaw",
    "storage chest",
    "chiffonier",
    "bell or wind chime",
    "china cabinet",
    "Christmas stocking",
    "church",
    "movie theater",
    "cleaver",
    "cliff dwelling",
    "cloak",
    "clogs",
    "cocktail shaker",
    "coffee mug",
    "coffeemaker",
    "spiral or coil",
    "combination lock",
    "computer keyboard",
    "candy store",
    "container ship",
    "convertible",
    "corkscrew",
    "cornet",
    "cowboy boot",
    "cowboy hat",
    "cradle",
    "construction crane",
    "crash helmet",
    "crate",
    "infant bed",
    "Crock Pot",
    "croquet ball",
    "crutch",
    "cuirass",
    "dam",
    "desk",
    "desktop computer",
    "rotary dial telephone",
    "diaper",
    "digital clock",
    "digital watch",
    "dining table",
    "dishcloth",
    "dishwasher",
    "disc brake",
    "dock",
    "dog sled",
    "dome",
    "doormat",
    "drilling rig",
    "drum",
    "drumstick",
    "dumbbell",
    "Dutch oven",
    "electric fan",
    "electric guitar",
    "electric locomotive",
    "entertainment center",
    "envelope",
    "espresso machine",
    "face powder",
    "feather boa",
    "filing cabinet",
    "fireboat",
    "fire truck",
    "fire screen",
    "flagpole",
    "flute",
    "folding chair",
    "football helmet",
    "forklift",
    "fountain",
    "fountain pen",
    "four-poster bed",
    "freight car",
    "French horn",
    "frying pan",
    "fur coat",
    "garbage truck",
    "gas mask or respirator",
    "gas pump",
    "goblet",
    "go-kart",
    "golf ball",
    "golf cart",
    "gondola",
    "gong",
    "gown",
    "grand piano",
    "greenhouse",
    "radiator grille",
    "grocery store",
    "guillotine",
    "hair clip",
    "hair spray",
    "half-track",
    "hammer",
    "hamper",
    "hair dryer",
    "hand-held computer",
    "handkerchief",
    "hard disk drive",
    "harmonica",
    "harp",
    "combine harvester",
    "hatchet",
    "holster",
    "home theater",
    "honeycomb",
    "hook",
    "hoop skirt",
    "gymnastic horizontal bar",
    "horse-drawn vehicle",
    "hourglass",
    "iPod",
    "clothes iron",
    "carved pumpkin",
    "jeans",
    "jeep",
    "T-shirt",
    "jigsaw puzzle",
    "rickshaw",
    "joystick",
    "kimono",
    "knee pad",
    "knot",
    "lab coat",
    "ladle",
    "lampshade",
    "laptop computer",
    "lawn mower",
    "lens cap",
    "letter opener",
    "library",
    "lifeboat",
    "lighter",
    "limousine",
    "ocean liner",
    "lipstick",
    "slip-on shoe",
    "lotion",
    "music speaker",
    "loupe magnifying glass",
    "sawmill",
    "magnetic compass",
    "messenger bag",
    "mailbox",
    "tights",
    "one-piece bathing suit",
    "manhole cover",
    "maraca",
    "marimba",
    "mask",
    "matchstick",
    "maypole",
    "maze",
    "measuring cup",
    "medicine cabinet",
    "megalith",
    "microphone",
    "microwave oven",
    "military uniform",
    "milk can",
    "minibus",
    "miniskirt",
    "minivan",
    "missile",
    "mitten",
    "mixing bowl",
    "mobile home",
    "ford model t",
    "modem",
    "monastery",
    "monitor",
    "moped",
    "mortar and pestle",
    "graduation cap",
    "mosque",
    "mosquito net",
    "vespa",
    "mountain bike",
    "tent",
    "computer mouse",
    "mousetrap",
    "moving van",
    "muzzle",
    "metal nail",
    "neck brace",
    "necklace",
    "baby pacifier",
    "notebook computer",
    "obelisk",
    "oboe",
    "ocarina",
    "odometer",
    "oil filter",
    "pipe organ",
    "oscilloscope",
    "overskirt",
    "bullock cart",
    "oxygen mask",
    "product packet / packaging",
    "paddle",
    "paddle wheel",
    "padlock",
    "paintbrush",
    "pajamas",
    "palace",
    "pan flute",
    "paper towel",
    "parachute",
    "parallel bars",
    "park bench",
    "parking meter",
    "railroad car",
    "patio",
    "payphone",
    "pedestal",
    "pencil case",
    "pencil sharpener",
    "perfume",
    "Petri dish",
    "photocopier",
    "plectrum",
    "Pickelhaube",
    "picket fence",
    "pickup truck",
    "pier",
    "piggy bank",
    "pill bottle",
    "pillow",
    "ping-pong ball",
    "pinwheel",
    "pirate ship",
    "drink pitcher",
    "block plane",
    "planetarium",
    "plastic bag",
    "plate rack",
    "farm plow",
    "plunger",
    "Polaroid camera",
    "pole",
    "police van",
    "poncho",
    "pool table",
    "soda bottle",
    "plant pot",
    "potter's wheel",
    "power drill",
    "prayer rug",
    "printer",
    "prison",
    "missile",
    "projector",
    "hockey puck",
    "punching bag",
    "purse",
    "quill",
    "quilt",
    "race car",
    "racket",
    "radiator",
    "radio",
    "radio telescope",
    "rain barrel",
    "recreational vehicle",
    "fishing casting reel",
    "reflex camera",
    "refrigerator",
    "remote control",
    "restaurant",
    "revolver",
    "rifle",
    "rocking chair",
    "rotisserie",
    "eraser",
    "rugby ball",
    "ruler measuring stick",
    "sneaker",
    "safe",
    "safety pin",
    "salt shaker",
    "sandal",
    "sarong",
    "saxophone",
    "scabbard",
    "weighing scale",
    "school bus",
    "schooner",
    "scoreboard",
    "CRT monitor",
    "screw",
    "screwdriver",
    "seat belt",
    "sewing machine",
    "shield",
    "shoe store",
    "shoji screen / room divider",
    "shopping basket",
    "shopping cart",
    "shovel",
    "shower cap",
    "shower curtain",
    "ski",
    "balaclava ski mask",
    "sleeping bag",
    "slide rule",
    "sliding door",
    "slot machine",
    "snorkel",
    "snowmobile",
    "snowplow",
    "soap dispenser",
    "soccer ball",
    "sock",
    "solar thermal collector",
    "sombrero",
    "soup bowl",
    "keyboard space bar",
    "space heater",
    "space shuttle",
    "spatula",
    "motorboat",
    "spider web",
    "spindle",
    "sports car",
    "spotlight",
    "stage",
    "steam locomotive",
    "through arch bridge",
    "steel drum",
    "stethoscope",
    "scarf",
    "stone wall",
    "stopwatch",
    "stove",
    "strainer",
    "tram",
    "stretcher",
    "couch",
    "stupa",
    "submarine",
    "suit",
    "sundial",
    "sunglasses",
    "sunglasses",
    "sunscreen",
    "suspension bridge",
    "mop",
    "sweatshirt",
    "swim trunks / shorts",
    "swing",
    "electrical switch",
    "syringe",
    "table lamp",
    "tank",
    "tape player",
    "teapot",
    "teddy bear",
    "television",
    "tennis ball",
    "thatched roof",
    "front curtain",
    "thimble",
    "threshing machine",
    "throne",
    "tile roof",
    "toaster",
    "tobacco shop",
    "toilet seat",
    "torch",
    "totem pole",
    "tow truck",
    "toy store",
    "tractor",
    "semi-trailer truck",
    "tray",
    "trench coat",
    "tricycle",
    "trimaran",
    "tripod",
    "triumphal arch",
    "trolleybus",
    "trombone",
    "hot tub",
    "turnstile",
    "typewriter keyboard",
    "umbrella",
    "unicycle",
    "upright piano",
    "vacuum cleaner",
    "vase",
    "vaulted or arched ceiling",
    "velvet fabric",
    "vending machine",
    "vestment",
    "viaduct",
    "violin",
    "volleyball",
    "waffle iron",
    "wall clock",
    "wallet",
    "wardrobe",
    "military aircraft",
    "sink",
    "washing machine",
    "water bottle",
    "water jug",
    "water tower",
    "whiskey jug",
    "whistle",
    "hair wig",
    "window screen",
    "window shade",
    "Windsor tie",
    "wine bottle",
    "airplane wing",
    "wok",
    "wooden spoon",
    "wool",
    "split-rail fence",
    "shipwreck",
    "sailboat",
    "yurt",
    "website",
    "comic book",
    "crossword",
    "traffic or street sign",
    "traffic light",
    "dust jacket",
    "menu",
    "plate",
    "guacamole",
    "consomme",
    "hot pot",
    "trifle",
    "ice cream",
    "popsicle",
    "baguette",
    "bagel",
    "pretzel",
    "cheeseburger",
    "hot dog",
    "mashed potatoes",
    "cabbage",
    "broccoli",
    "cauliflower",
    "zucchini",
    "spaghetti squash",
    "acorn squash",
    "butternut squash",
    "cucumber",
    "artichoke",
    "bell pepper",
    "cardoon",
    "mushroom",
    "Granny Smith apple",
    "strawberry",
    "orange",
    "lemon",
    "fig",
    "pineapple",
    "banana",
    "jackfruit",
    "cherimoya (custard apple)",
    "pomegranate",
    "hay",
    "carbonara",
    "chocolate syrup",
    "dough",
    "meatloaf",
    "pizza",
    "pot pie",
    "burrito",
    "red wine",
    "espresso",
    "tea cup",
    "eggnog",
    "mountain",
    "bubble",
    "cliff",
    "coral reef",
    "geyser",
    "lakeshore",
    "promontory",
    "sandbar",
    "beach",
    "valley",
    "volcano",
    "baseball player",
    "bridegroom",
    "scuba diver",
    "rapeseed",
    "daisy",
    "yellow lady's slipper",
    "corn",
    "acorn",
    "rose hip",
    "horse chestnut seed",
    "coral fungus",
    "agaric",
    "gyromitra",
    "stinkhorn mushroom",
    "earth star fungus",
    "hen of the woods mushroom",
    "bolete",
    "corn cob",
    "toilet paper",
]


def resize(image, image_size, center_crop=False):
    # image is a numpy array with shape (channels, height, width), reverse to (height, width, channels)
    image = image.numpy().transpose(1, 2, 0)

    if len(image.shape) == 3 and image.shape[-1] == 4:
        # alpha matting with white background
        image = (image * 255).astype(np.uint8)
        alpha = image[:, :, 3, np.newaxis]
        image = alpha / 255.0 * image[:, :, :3] + 255 - alpha
        image = np.rint(image.clip(min=0, max=255)).astype(np.uint8)
        image = (image / 255.0).astype(np.float32)

    h, w, _ = image.shape
    if not (h == image_size and w == image_size):
        if center_crop:
            # Resize keeping the aspect ratio, shortest side resized to image_size
            if h > w:
                oh, ow = image_size * h // w, image_size
            else:
                oh, ow = image_size, image_size * w // h
            image = cv2.resize(image, (ow, oh), interpolation=cv2.INTER_AREA)

            # Center crop to final image size
            i = (oh - image_size) // 2
            j = (ow - image_size) // 2
            image = image[i : i + image_size, j : j + image_size]
        else:
            # Resize to image_size x image_size
            image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)

    # normalize
    image = (image - 0.5) / 0.5
    return image


if __name__ == "__main__":
    assert jax.local_device_count() == 8

    # load dataset
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)
    transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Lambda(lambda x: resize(x, 256)),
        ]
    )
    ds = load_dataset(transform=transforms)
    every_n = int(1 / args.fraction)
    dl = torch.utils.data.DataLoader(ds, batch_size=every_n, num_workers=20, shuffle=False)

    # paths
    tokenizer_name = args.tokenizer_name
    train_run = args.train_run

    # load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # get model
    print("Loading model...")
    wandb_api = wandb.Api()
    train_run = wandb_api.run(train_run)
    entity = train_run.entity
    project = train_run.project
    model_path = train_run.config["output_dir"]

    # model
    config = load_config(f"{model_path}/config.json")
    model = CLIPModel(**config)
    rng = jax.random.PRNGKey(0)
    logical_shape = jax.eval_shape(lambda rng: model.init_weights(rng), rng)["params"]

    # create mesh
    dev_mesh = create_device_mesh((jax.local_device_count(), 1))
    mesh = Mesh(dev_mesh, ("data", "model"))
    data_spec = PartitionSpec("data")

    # init params
    @partial(pjit, in_shardings=(), out_shardings=None)
    def init_params():
        return jax.tree_map(lambda x: jnp.zeros(x.shape, dtype=x.dtype), logical_shape)

    with mesh:
        params = init_params()

    # get checkpoints
    available_steps = checkpoints.available_steps(model_path)
    if args.latest_only:
        available_steps = available_steps[-1:]

    # unique wandb run per model
    wandb_id = f"eval_{train_run.id}"
    wandb_run_name = f"Eval {train_run.name}"

    # get last step logged
    try:
        last_step_logged = wandb_api.run(f"{entity}/{project}/{wandb_id}").summary.get("state/step", 0)
    except:
        last_step_logged = 0

    @partial(pjit, in_shardings=(data_spec, data_spec, None, None, None, None, None), out_shardings=(None, None))
    def get_scores(input_ids, attention_mask, pixel_values, labels, label_mask, ground_truth, params):
        assert pixel_values.shape[0] == 1, "only support 1 image at a time"
        encoder_outputs = model.apply({"params": params}, pixel_values=pixel_values, method=model.get_image_features)[
            "vision_model_output"
        ]["last_hidden_state"]
        encoder_hidden_states = jnp.repeat(encoder_outputs, input_ids.shape[0], axis=0)
        encoder_hidden_states = with_sharding_constraint(encoder_hidden_states, data_spec)
        logits = model.apply(
            {"params": params},
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            decode=False,
            method=model.get_text_features,
        )["text_model_output"]["last_hidden_state"]
        score = -optax.softmax_cross_entropy_with_integer_labels(logits, labels) * label_mask
        score = score.sum(axis=-1)
        top_k = score.argsort()[::-1][:5]
        acc_top_1 = (top_k[0] == ground_truth).sum()
        acc_top_5 = (top_k == ground_truth).sum()
        return acc_top_1, acc_top_5

    # start run
    run = wandb.init(
        id=wandb_id,
        name=wandb_run_name,
        resume="allow",
        entity=entity,
        project=project,
        job_type="eval",
    )

    # get text inputs
    captions = ds.classes

    txt_inputs = tokenizer(
        captions,
        padding="max_length",
        truncation=True,
        max_length=config["text_config"]["max_length"],
        return_tensors="np",
    )
    labels = shift_tokens_left(txt_inputs["input_ids"], pad_token_id=tokenizer.pad_token_id)
    labels_mask = shift_tokens_left(txt_inputs["attention_mask"], pad_token_id=0)

    for step in available_steps:
        if step <= last_step_logged:
            continue

        # restore checkpoint
        print(f"Restoring checkpoint at step {step}...")
        ckpt = {"params": params}
        restore_args = orbax_utils.restore_args_from_target(ckpt)
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        orbax_options = orbax.checkpoint.CheckpointManagerOptions()
        checkpoint_manager = orbax.checkpoint.CheckpointManager(model_path, orbax_checkpointer, orbax_options)
        ckpt = checkpoint_manager.restore(step, ckpt, restore_kwargs={"restore_args": restore_args, "transforms": {}})
        params = ckpt["params"]

        # calculate metrics
        acc_top_1_all = []
        acc_top_5_all = []
        n_items = []

        # iterate
        for i, batch in enumerate(tqdm(dl)):
            images, ground_truths = batch
            if every_n > 1:
                images = images[:1]
                ground_truths = ground_truths[:1]
            pixel_values = jnp.asarray(images)
            ground_truths = jnp.asarray(ground_truths)
            with mesh:
                acc_top_1, acc_top_5 = get_scores(
                    txt_inputs["input_ids"],
                    txt_inputs["attention_mask"],
                    pixel_values,
                    labels,
                    labels_mask,
                    ground_truths,
                    params,
                )
            acc_top_1_all.append(acc_top_1)
            acc_top_5_all.append(acc_top_5)
            n_items.append(len(images))

        # weighted average
        acc_top_1 = np.average(acc_top_1_all, weights=n_items)
        acc_top_5 = np.average(acc_top_5_all, weights=n_items)
        print(f"acc_top_1: {acc_top_1:.4f}")
        print(f"acc_top_5: {acc_top_5:.4f}")

        # log metrics
        run.log(
            {
                "state/step": step,
                "eval/acc_top_1": acc_top_1,
                "eval/acc_top_5": acc_top_5,
            }
        )
