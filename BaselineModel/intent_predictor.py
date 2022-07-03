class IntentPredictor:
    def __init__(self,
                 tasks = None):

        self.tasks = tasks

    def predict_intent(self,
                       input: str):
        raise NotImplementedError
    
class EmbKnnIntentPredictor(IntentPredictor):
    def __init__(self,
                 model,
                 tasks = None):
    
        super().__init__(tasks)
        
        self.model = model

    def predict_intent(self,
                       input: str):

        if self.model.cached_embeddings is None:
            example_sentences = []
            for t in self.tasks:
                for e in t['examples']:
                    example_sentences.append(e)
            self.model.cache(example_sentences)
        
        results = self.model.predict([input])[0]
        maxScore, maxIndex = results.max(dim = 0)
        
        maxScore = maxScore.item()
        maxIndex = maxIndex.item()

        index = -1
        for t in self.tasks:
            for e in t['examples']:
                index += 1

                if index == maxIndex:
                    intent = t['task']
                    matched_example = e

        return intent, maxScore, matched_example
