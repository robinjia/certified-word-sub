"""Some tests."""
import argparse
import sys
import torch
import torch.nn.functional as F

import ibp

def test_logsoftmax():
  x = torch.tensor([[0.0, 1.0, 2.0], [0.0, 2.0, 4.0]])
  print('x: ', x)
  print('F.logsoftmax(x, dim=0): ', F.log_softmax(x, dim=0))
  x_ibp = ibp.IntervalBoundedTensor(x, x, x)
  ls = ibp.log_softmax(x_ibp, dim=0)
  print('ibp.log_softmax(x, lb=x, ub=x): ', ls.val, ls.lb, ls.ub)

  lb = x - torch.tensor(0.1)
  ub = x + torch.tensor(0.1)
  print('lb: ', lb)
  print('ub: ', lb)
  x_ibp = ibp.IntervalBoundedTensor(x, lb, ub)
  ls = ibp.log_softmax(x_ibp, dim=0)
  print('ibp.log_softmax(x, lb, ub): ', ls.val, ls.lb, ls.ub)

def test_bmm():
  m1 = torch.tensor([[-1, 2], [3, 2], [-3, 1]], dtype=torch.float).view(1, 3, 2)
  m2 = torch.tensor([[4, 5], [-4, -5]], dtype=torch.float).view(1, 2, 2)
  z = ibp.bmm(ibp.IntervalBoundedTensor(m1, m1, m1), ibp.IntervalBoundedTensor(m2, m2, m2))
  m1_bounded = ibp.IntervalBoundedTensor(m1, m1 - torch.tensor(0.1), m1 + torch.tensor(0.1))
  m2_bounded = ibp.IntervalBoundedTensor(m2, m2 - torch.tensor(0.1), m2 + torch.tensor(0.1))
  print('ibp.bmm, exact:', z.val, z.lb, z.ub)
  z2 = ibp.bmm(m1_bounded, m2_bounded)
  print('ibp.bmm, bound both:', z2.val, z2.lb, z2.ub)
  z3 = ibp.bmm(m1_bounded, m2)
  print('ibp.bmm, bound first:', z3.val, z3.lb, z3.ub)
  z4 = ibp.bmm(m1, m2_bounded)
  print('ibp.bmm, bound second:', z4.val, z4.lb, z4.ub)

  

def main():
  test_bmm()
  test_logsoftmax()

if __name__ == '__main__':
  main()

