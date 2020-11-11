// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bpetokenizer

import (
	"fmt"
	"github.com/nlpodyssey/spago/pkg/nlp/tokenizers"
	"github.com/nlpodyssey/spago/pkg/nlp/tokenizers/models"
	"github.com/nlpodyssey/spago/pkg/nlp/tokenizers/models/bpemodel"
	"github.com/nlpodyssey/spago/pkg/nlp/tokenizers/normalizers/normalizedstring"
	"github.com/nlpodyssey/spago/pkg/nlp/tokenizers/pretokenizers/bytelevelpretokenizer"
)

var _ tokenizers.Tokenizer = &BPETokenizer{}

// BPETokenizer is a higher-level tokenizer, which includes byte-level pre-tokenization.
type BPETokenizer struct {
	preTokenizer *bytelevelpretokenizer.ByteLevelPreTokenizer
	model        *bpemodel.BpeModel
}

// New returns a new BPETokenizer.
func New(
	preTokenizer *bytelevelpretokenizer.ByteLevelPreTokenizer,
	model *bpemodel.BpeModel,
) *BPETokenizer {
	return &BPETokenizer{
		preTokenizer: preTokenizer,
		model:        model,
	}
}

// Tokenize performs byte-level pre-tokenization and BPE tokenization.
func (t *BPETokenizer) Tokenize(text string) []tokenizers.StringOffsetsPair {
	ns := normalizedstring.NewNormalizedString(text)
	preTokens, err := t.preTokenizer.PreTokenize(ns)
	if err != nil {
		panic(fmt.Sprintf("BPETokenizer PreTokenize error: %v", err))
	}

	tokens, err := t.model.Tokenize(preTokens)
	if err != nil {
		panic(fmt.Sprintf("BPETokenizer Tokenize error: %v", err))
	}

	return tokensToStringOffsetsPairs(tokens)
}

func tokensToStringOffsetsPairs(tokens []models.Token) []tokenizers.StringOffsetsPair {
	sop := make([]tokenizers.StringOffsetsPair, len(tokens))

	for i, token := range tokens {
		sop[i] = tokenizers.StringOffsetsPair{
			String: token.Value,
			Offsets: tokenizers.OffsetsType{
				Start: token.Offsets.Start,
				End:   token.Offsets.End,
			},
		}
	}

	return sop
}
