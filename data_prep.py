"""
Synthetic data generation and preprocessing for Fake News Detection.

This module generates realistic fake and real news samples for training
and provides data preparation functions.
"""

import random
from utils import preprocess_text


class SyntheticDataGenerator:
    """
    Generate synthetic fake and real news samples for model training.

    Uses template-based generation with randomization to create realistic
    and diverse fake/real news samples.
    """

    ENTITIES = [
        "Apple", "Microsoft", "Tesla", "Google", "Amazon", "Meta",
        "Donald Trump", "Joe Biden", "Elon Musk", "Bill Gates",
        "Taylor Swift", "Kim Kardashian", "Oprah Winfrey",
        "WHO", "CDC", "FDA", "NASA", "Pentagon", "FBI", "CIA"
    ]

    TOPICS = ["politics", "health", "technology", "business", "science", "sports"]

    ADJECTIVES = ["shocking", "exclusive", "leaked", "breaking", "classified", "viral"]

    ACTIONS = {
        "politics": ["announces", "reveals", "claims", "exposes", "confirms", "denies"],
        "health": ["cures", "causes", "prevents", "leads to", "triggers", "affects"],
        "technology": ["launches", "releases", "develops", "creates", "unveils", "introduces"],
        "business": ["acquires", "merges", "reports", "files", "announces", "expands"],
        "science": ["discovers", "proves", "demonstrates", "shows", "confirms", "reveals"],
        "sports": ["wins", "signs", "retires", "scores", "breaks", "achieves"]
    }

    EFFECTS = {
        "politics": ["controversy", "scandal", "uproar", "panic", "chaos", "turmoil"],
        "health": ["death", "illness", "epidemics", "side effects", "cancer", "mutations"],
        "technology": ["revolution", "disruption", "change", "breakthrough", "failure", "risk"],
        "business": ["collapse", "growth", "success", "bankruptcy", "profit", "loss"],
        "science": ["breakthrough", "danger", "change", "progress", "crisis", "hope"],
        "sports": ["retirement", "trade", "contract", "record", "upset", "victory"]
    }

    SOURCES = [
        "Reuters", "BBC", "AP News", "CNN", "The Guardian",
        "Associated Press", "researchers", "scientists",
        "officials", "government", "experts"
    ]

    METHODS = ["study", "research", "investigation", "analysis", "survey", "experiment"]

    PERCENTAGES = ["90%", "85%", "75%", "65%", "99%", "80%", "70%", "60%"]

    FAKE_TEMPLATES = [
        "SHOCKING: {entity} {action} causing {effect}! This will blow your mind!",
        "EXCLUSIVE: {entity} secretly {action} to {effect} citizens!",
        "{entity} claims {percentage} of {topic} are {effect} - UNCONFIRMED!",
        "BREAKING: {influencer} EXPOSES {entity} - You won't believe what happened!",
        "{entity} allegedly {action} - sources say major {effect} coming!",
        "LEAKED: Inside sources reveal {entity} {action} {effect} - cover up attempt!",
        "WARNING: New {entity} {action} causing widespread {effect}!",
        "{entity} warned us but nobody listened - {effect} now spreading!",
        "MASSIVE: {entity} {action} but mainstream media WON'T cover it!",
        "CONSPIRACY: {entity} secretly working to {effect} - evidence hidden!",
        "{entity} just announced {action} and it's causing mass {effect}!",
        "ALERT: {entity} {action} - {percentage} reported {effect}!",
        "Nobody is talking about what {entity} did - total {effect}!",
        "SCANDAL: {entity} caught {action} - massive {effect} expected!",
        "UNBELIEVABLE: {entity} claims {action} will lead to {effect}!",
        "Doctors HATE this - {entity} reveals {action} for {effect}!",
    ]

    REAL_TEMPLATES = [
        "{entity} announced {action} according to official sources.",
        "A recent {method} shows that {entity} {action} leading to {effect}.",
        "{entity} released a statement saying {action} effective next month.",
        "{source} reports {entity} {action} in a major development.",
        "Scientists report that {entity} {action} based on extensive {method}.",
        "{entity} officially {action}, confirmed by multiple sources.",
        "New {method} from {source} indicates {entity} {action}.",
        "{entity} {action}, the company announced in a press release.",
        "Research demonstrates that {entity} {action} with significant impact.",
        "{entity} confirmed to {action}, according to recent reports.",
        "Government officials state that {entity} will {action}.",
        "{entity} {action} following comprehensive {method} and review.",
        "Multiple sources confirm {entity} has {action}.",
        "{source} released data showing {entity} {action}.",
        "{entity} makes announcement: will {action} next quarter.",
        "Experts confirm {entity} {action} based on available evidence.",
    ]

    def __init__(self, random_seed=42):
        random.seed(random_seed)

    def generate_dataset(self, samples_per_class=1000):
        texts = []
        labels = []

        for _ in range(samples_per_class):
            texts.append(self._generate_real_sample())
            labels.append(0)

        for _ in range(samples_per_class):
            texts.append(self._generate_fake_sample())
            labels.append(1)

        combined = list(zip(texts, labels))
        random.shuffle(combined)
        texts, labels = zip(*combined)

        return list(texts), list(labels)

    def _generate_fake_sample(self):
        template = random.choice(self.FAKE_TEMPLATES)
        topic = random.choice(self.TOPICS)

        action = random.choice(self.ACTIONS.get(topic, self.ACTIONS["politics"]))
        effect = random.choice(self.EFFECTS.get(topic, self.EFFECTS["politics"]))

        filled = template.format(
            entity=random.choice(self.ENTITIES),
            action=action,
            effect=effect,
            percentage=random.choice(self.PERCENTAGES),
            influencer=random.choice([x for x in self.ENTITIES if ' ' in x]),
            topic=topic
        )

        return filled

    def _generate_real_sample(self):
        template = random.choice(self.REAL_TEMPLATES)
        topic = random.choice(self.TOPICS)

        action = random.choice(self.ACTIONS.get(topic, self.ACTIONS["politics"]))
        effect = random.choice(self.EFFECTS.get(topic, self.EFFECTS["politics"]))

        filled = template.format(
            entity=random.choice(self.ENTITIES),
            action=action,
            effect=effect,
            source=random.choice(self.SOURCES),
            method=random.choice(self.METHODS)
        )

        return filled


def prepare_data_for_training(texts, labels):
    return [preprocess_text(text) for text in texts]


def split_data(texts, labels, train_size=0.8, random_seed=42):
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=(1 - train_size),
        random_state=random_seed,
        stratify=labels
    )

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    generator = SyntheticDataGenerator()
    texts, labels = generator.generate_dataset(samples_per_class=100)
    print(f"Generated {len(texts)} samples")
    print(f"Fake samples: {sum(1 for l in labels if l == 1)}")
    print(f"Real samples: {sum(1 for l in labels if l == 0)}")