// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mat

type Matrix interface {
	ZerosLike() Matrix
	OnesLike() Matrix
	Clone() Matrix
	Copy(other Matrix)
	Zeros()
	Dims() (r, c int)
	Rows() int
	Columns() int
	Size() int
	LastIndex() int
	Data() []float64
	IsVector() bool
	IsScalar() bool
	Scalar() float64
	Set(i int, j int, v float64)
	At(i int, j int) float64
	SetVec(i int, v float64)
	AtVec(i int) float64
	T() Matrix
	Reshape(r, c int) Matrix
	Apply(fn func(i, j int, v float64) float64, a Matrix)
	ApplyWithAlpha(fn func(i, j int, v float64, alpha ...float64) float64, a Matrix, alpha ...float64)
	AddScalar(n float64) Matrix
	AddScalarInPlace(n float64) Matrix
	SubScalar(n float64) Matrix
	SubScalarInPlace(n float64) Matrix
	ProdScalar(n float64) Matrix
	ProdScalarInPlace(n float64) Matrix
	ProdMatrixScalarInPlace(m Matrix, n float64) Matrix
	Add(other Matrix) Matrix
	AddInPlace(other Matrix) Matrix
	Sub(other Matrix) Matrix
	SubInPlace(other Matrix) Matrix
	Prod(other Matrix) Matrix
	ProdInPlace(other Matrix) Matrix
	Div(other Matrix) Matrix
	DivInPlace(other Matrix) Matrix
	Mul(other Matrix) Matrix
	DotUnitary(other Matrix) float64
	Pow(power float64) Matrix
	Norm(pow float64) float64
	Sqrt() Matrix
	ClipInPlace(min, max float64) Matrix
	Abs() Matrix
	Sum() float64
	Max() float64
	Min() float64
	String() string
	SetData(data []float64)
}

func ConcatV(vs ...Matrix) Matrix {
	cup := 0
	for _, v := range vs {
		cup += v.Size()
	}
	data := make([]float64, 0, cup)
	for _, v := range vs {
		if !v.IsVector() {
			panic("mat: required vector, found matrix")
		}
		data = append(data, v.Data()...)
	}
	return NewVecDense(data)
}

func ConcatH(ms ...Matrix) *Dense {
	rows := len(ms)
	cols := ms[0].Rows()
	out := NewEmptyDense(rows, cols)
	for i, x := range ms {
		for j := 0; j < cols; j++ {
			out.Set(i, j, x.At(j, 0))
		}
	}
	return out
}
