import gym
import time
from .wikienv import WikiEnv

class E5WikiEnv(WikiEnv):
    def __init__(self, retriever, k=5):
        super().__init__()
        self.retriever = retriever
        self.k = k

    def search_step(self, entity):
        old_time = time.time()
        # [Unified] Use retriever.search
        # Note: entity might come with search[...] wrapper handled by step(), here we get raw entity
        # ReAct agent sends "search[query]". wikienv strips it.
        
        # retriever.search returns list of dicts: [{'title':..., 'contents':..., 'score':...}]
        results = self.retriever.search(entity, k=self.k)
        
        self.search_time += time.time() - old_time
        self.num_searches += 1
        
        if not results:
            self.obs = f"Could not find {entity}. Similar: []"
        else:
            # ReAct expects a page content or list of results?
            # Original ReAct usually works on a "Page".
            # But E5 retriever returns snippets.
            # If we want to support "lookup", we need a "page".
            # For open-domain RAG, we often return concatenation of top-k results as the "page".
            # Or we return the snippets directly.
            
            # Construct a virtual page from results
            self.page = ""
            self.result_titles = []
            for res in results:
                title = res.get('title', '')
                content = res.get('contents', '') or res.get('text', '')
                self.result_titles.append(title)
                self.page += f"Title: {title}\nContent: {content}\n\n"
            
            # Update observation
            # Standard ReAct behavior: "search" returns the content or summary?
            # WikiEnv.search_step sets self.page and self.obs.
            # self.obs = self.get_page_obs(self.page) -> first 5 sentences.
            
            self.obs = self.get_page_obs(self.page)
            self.lookup_keyword = self.lookup_list = self.lookup_cnt = None

    def step(self, action):
        # Override step to capture is_poisoned metadata if available (for attack logging)
        obs, reward, done, info = super().step(action)
        
        # Check if the last search results contained poisoned data
        # Only feasible if we store the last results metadata in search_step
        # But WikiEnv doesn't store full metadata. 
        # For now, let's keep it simple. AttackManager injection handles poisoning.
        # If we need logging, we might need to enhance this.
        return obs, reward, done, info
