import pytest
import torch
from safegen import SafeGen, create_safegen

class TestSafeGen:
    def test_init(self):
        m = SafeGen(dim=256, hidden_dim=512)
        assert m.dim == 256
    
    def test_forward(self):
        m = SafeGen(dim=256)
        x = torch.randn(32, 256)
        assert m(x).shape == (32, 256)
    
    def test_factory(self):
        m = create_safegen()
        assert isinstance(m, SafeGen)

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
