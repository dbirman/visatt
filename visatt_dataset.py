class VisAttDataset(Dataset):
    """
    This dataset contains a list of numbers in the range [a,b] inclusive
    """
    def __init__(self, folder='~/proj/visatt/data/out/'):
        super(MyDataset, self).__init__()
        
        self.folder = folder
        
    def __len__(self):
        return self.b - self.a + 1
        
    def __getitem__(self, index):
        assert self.a <= index <= self.b
        
        return index, index**2