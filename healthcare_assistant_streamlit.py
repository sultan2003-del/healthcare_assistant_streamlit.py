from sentence_transformers import SentenceTransformer, util
import torch

class FileReader:
    def __init__(self, file_path):
        self.file_path = file_path

    def read_data(self):
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                return [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            return ["Error: Health information document not found."]

class AnsweringAgent:
    def __init__(self, knowledge_base):
        self.knowledge_base = knowledge_base
        self.model = SentenceTransformer('all-MiniLM-L6-v2')  # Pre-trained model for sentence embeddings
        # Precompute the embeddings for the knowledge base
        self.embeddings = self.model.encode(self.knowledge_base, convert_to_tensor=True)

    def generate_answer(self, question):
        question_embedding = self.model.encode(question, convert_to_tensor=True)
        cosine_scores = util.cos_sim(question_embedding, self.embeddings)  # Compute cosine similarity
        
        # Find the best match (highest cosine similarity)
        top_result = torch.topk(cosine_scores, k=1)
        top_index = top_result.indices.item()

        # Return the best snippet if the similarity is strong enough
        best_snippet = self.knowledge_base[top_index]

        if top_result.values.item() > 0.3:  # If the match is strong enough
            return best_snippet
        else:
            return "I'm sorry, I couldn't find a relevant answer to your question."

class HealthcareAssistantSystem:
    def __init__(self, health_info_file):
        self.reader = FileReader(health_info_file)
        self.answering_agent = None

    def setup(self):
        knowledge_base = self.reader.read_data()
        self.answering_agent = AnsweringAgent(knowledge_base)

    def handle_user_question(self, question):
        answer = self.answering_agent.generate_answer(question)
        return answer


if __name__ == "__main__":
    health_info_file = 'health_info.txt'  # Replace with the correct path to your file
    assistant = HealthcareAssistantSystem(health_info_file)
    assistant.setup()

    print("Healthcare Assistant is ready! (Type 'exit' to quit)")
    while True:
        user_question = input("\nAsk your health question: ")
        if user_question.lower() in ["exit", "quit"]:
            break
        response = assistant.handle_user_question(user_question)
        print(f"\nAnswer: {response}")
