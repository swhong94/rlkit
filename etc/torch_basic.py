
#numpy와 비슷한 기능을 하는 torch를 사용하여 코드를 작성할 수 있다.
#torch는 numpy와 비슷한 기능을 제공하며, GPU를 사용하여 연산을 수행할 수 있다.
import torch

a = torch.tensor([1, 2, 3, 4])
print(a)
print(type(a))
print(a.dtype)
print(a.shape)
b=torch.tensor([[1, 2, 3.1, 4]])
print(b.dtype)
print(b)


A=torch.tensor([[1, 2, 3], [4, 5, 6]])
print(A)
print(A.shape)
print(A.ndim)   #차원
print(A.numel())    #원소의 개수
print(A.size(0))    #행의 개수
print(A.size(1))    #열의 개수
A=torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
print(A)
#A=torch.tensor([[1, 2], [4, 5, 6]])
#에러가 나는 이유: 텐서의 모든 행은 같은 길이를 가져야 한다. 행렬이기 때문에
print(A)


print(torch.zeros(2, 3))
print(torch.zeros(5))
#print(torch.zeros_like(A))
print(torch.ones(4))
print(torch.arange(3,10,2))
print(torch.arange(0,1,0.1))
print(torch.linspace(0,1,5))

a=torch.tensor([1, 2, 3])
b=torch.tensor([4, 5, 6])
print(a+b)
print(a-b)

print(a*b) #element-wise multiplication
print(a/b)  #element-wise division
print(a**2) #element-wise exponentiation
print(a**b) #element-wise exponentiation    '


A=torch.tensor([[1, 2, 3], [4, 5, 6]])
B=torch.tensor([[1, 2], [3, 4], [5, 6]])
print(A@B)  #행렬 곱은 @로 표현! #2x3 3x2 = 2x2
print(A.mm(B))  #행렬 곱
print(A.matmul(B))  #행렬 곱


