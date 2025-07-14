import os
import numpy as np

class DescriptorManager:
    def __init__(self, descriptor_dir='descriptors'):
        self.descriptor_dir = descriptor_dir
        self.available_methods = {}
        self.loaded_descriptors = {
            'sift': [],
            'orb': [],
            'freak': []
        }
        self.load_available_methods()
    
    def load_available_methods(self):
        if not os.path.exists(self.descriptor_dir):
            print(f"Descriptor directory {self.descriptor_dir} not found")
            return
        
        for method in ['sift', 'orb', 'freak']:
            path = os.path.join(self.descriptor_dir, f"{method}_descriptors.npy")
            if os.path.exists(path):
                self.available_methods[method] = path
        
        print(f"Found {len(self.available_methods)} available descriptor methods: {', '.join(self.available_methods.keys())}")
    
    def get_available_methods(self):
        return list(self.available_methods.keys())
    
    def load_method(self, method):
        if method.lower() not in self.available_methods:
            print(f"Method {method} not available")
            return []
        
        file_path = self.available_methods[method.lower()]
        try:
            descriptors = np.load(file_path, allow_pickle=True)
            descriptors = descriptors.tolist() 
            self.loaded_descriptors[method.lower()] = descriptors
            print(f"Loaded {len(descriptors)} {method.upper()} descriptor sets")
            return descriptors
        except Exception as e:
            print(f"Error loading {method} descriptors: {e}")
            return []
    
    def get_descriptors(self, method=None):
        if method:
            return self.loaded_descriptors.get(method.lower(), [])
        return self.loaded_descriptors