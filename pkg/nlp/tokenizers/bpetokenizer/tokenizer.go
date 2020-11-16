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
	"github.com/nlpodyssey/spago/pkg/nlp/tokenizers/vocabulary"
	"path/filepath"
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

const (
	defaultCacheCapacity           = 0
	defaultDropout                 = 0.0
	defaultUnknownToken            = ""
	defaultContinuingSubwordPrefix = ""
	defaultEndOfWordSuffix         = ""
	defaultPrefixSpaceEnabled      = false
)

// NewFromModelFolder returns a new BPETokenizer built from a
// pre-trained Roberta-compatible model, given the path to the
// folder containing the separate model and configuration files.
func NewFromModelFolder(path string) (*BPETokenizer, error) {
	vocabularyFilename := filepath.Join(path, "vocab.json")
	vocab, err := vocabulary.FromJSONFile(vocabularyFilename)
	if err != nil {
		return nil, fmt.Errorf("loading vocabulary from file %s: %v", vocabularyFilename, err)
	}

	mergesFilename := filepath.Join(path, "merges.txt")
	merges, err := bpemodel.MergeMapFromFile(
		mergesFilename,
		vocab,
		len(defaultContinuingSubwordPrefix),
	)
	if err != nil {
		return nil, fmt.Errorf("loading merges from file %s: %v", mergesFilename, err)
	}

	preTokenizer := bytelevelpretokenizer.NewByteLevelPreTokenizer(
		bytelevelpretokenizer.DefaultSplittingRegexp,
		defaultPrefixSpaceEnabled, // TODO: read from optional config?
	)

	model := bpemodel.NewBpeModel(
		vocab,
		merges,
		defaultCacheCapacity,
		defaultDropout,
		defaultUnknownToken,
		defaultContinuingSubwordPrefix,
		defaultEndOfWordSuffix,
	)

	return New(preTokenizer, model), nil
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
