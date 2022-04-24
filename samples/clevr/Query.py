class Query:

    def __init__(self, sentence: str):
        self.sentence = sentence
        self.words = sentence.split()

    def get_filters_by_actions(self):
        actions_and_words = self._group_words_by_action(self.words)

        actions_and_filters = []
        for (action, words_lst) in actions_and_words:
            objects_and_words = self._group_words_by_object(words_lst)

            object_filters = []
            for i in range(len(objects_and_words)):
                obj, words = objects_and_words[i]
                object_filters.append((obj, dict()))

            self._add_color_filters(objects_and_words, object_filters)
            self._add_relational_filters(objects_and_words, object_filters)
            actions_and_filters.append((action, object_filters))
        return actions_and_filters

    def _group_words_by_action(self, words):
        actions = ["select", "remove", "delete"]

        # We must split a sentence when we see a keyword
        # For example, consider the following two sentences:
        # (1) "Select all cubes left of the cylinder and the sphere." versus
        # (2) "Select all cubes left of the cylinder and select the sphere."
        # In (1), the "and" concatenates two constraints whereas in (2), the "and" concatenates two actions.

        actions_and_words = []

        # First, we find all action words.
        action_indices = []
        for i in range(len(words)):
            token = words[i]
            if token in actions:
                action_indices.append(i)

        # Then, we find all words corresponding to that action
        for k in range(len(action_indices)):
            i = action_indices[k]
            if k == len(action_indices) - 1:
                actions_and_words.append((words[i], words[i + 1:]))
            else:
                j = action_indices[k + 1]
                if words[j - 1] == "and":
                    j -= 1
                actions_and_words.append((words[i], words[i + 1:j]))
        return actions_and_words

    def _group_words_by_object(self, words):
        # words_doc = nlp(" ".join(words))
        # for token in words_doc:
        #     print(token.text, token.tag_, token.head.text, token.dep_)

        objects = ["cube", "sphere", "cylinder", "object", "thing", "shape", "one"]
        objects_and_words = []

        # First, we find all action words.
        object_indices = []
        for i in range(len(words)):
            raw_token = words[i]
            if raw_token[-1] == "s":  # Check if plural
                token = words[i][:-1]
            else:
                token = words[i]
            if token in objects:
                object_indices.append(i)

        # Then, we find all words corresponding to that object
        for k in range(len(object_indices)):
            i = object_indices[k]

            if words[i][-1] == "s":
                word = words[i][:-1]
            else:
                word = words[i]

            if k == 0:
                objects_and_words.append((word, words[:i]))
            else:
                j = object_indices[k - 1]
                if words[j + 1] == "and":
                    j += 1
                objects_and_words.append((word, words[j + 1:i]))
        return objects_and_words

    def _add_color_filters(self, objects_and_words, object_filters):

        colors = ["gray", "red", "blue", "green", "brown", "purple", "cyan", "yellow"]

        for i in range(len(objects_and_words)):
            obj, words = objects_and_words[i]

            for token in words:
                if token in colors:
                    if "color" in object_filters[i][1]:
                        object_filters[i][1]["color"].add(token)
                    else:
                        object_filters[i][1]["color"] = {token}

    def _add_relational_filters(self, objects_and_words, object_filters):

        relations = ["left", "right", "front", "behind"]

        for i in range(len(objects_and_words)):
            obj, words = objects_and_words[i]

            for token in words:
                if token in relations:
                    # Careful! Add relation to object 0 instead of current i
                    # considering the first element is the primary objective/target
                    # We also keep a reference to the referred object.
                    if "relation" in object_filters[0][1]:
                        object_filters[0][1]["relation"].add((token, i))
                    else:
                        object_filters[0][1]["relation"] = {(token, i)}
