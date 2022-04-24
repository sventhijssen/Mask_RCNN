from samples.clevr.KnowledgeBase import KnowledgeBase
from samples.clevr.Query import Query

sentences = [
    "select all green cubes",
    "select all green cubes and all spheres",
    "select the green cube the red sphere and the blue cylinders",
    "select the green cube the red sphere and the blue cylinder",
    "select all green cubes and select all spheres",
    "select all green cubes and delete all red spheres",
    "select all",
    "select everything",
    # "select all green cubes except for the one left of the red cylinder",  # Too difficult
    "select all objects",
    "select all green and blue objects",
    "select all shapes",
    "select all green and blue shapes",
    "select all things",
    "select all green and blue things",
    "select all green cubes left of the blue cylinder",
    "select all green cubes left of the blue cylinder and the red sphere",
    "select all green cubes left of the blue cylinder and right of the red sphere"
]

k = KnowledgeBase()

for sentence in sentences:
    query = Query(sentence)
    print(query.get_filters_by_actions())
