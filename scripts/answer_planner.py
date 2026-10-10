"""Question-only facet planning and observable source coverage (not entailment)."""
from __future__ import annotations

import re


def question_facets(question):
    """Split explicit compound requests without reading labels or paper identities."""
    question = re.sub(r'\s+', ' ', question).strip()
    pieces = re.split(r'\s+and\s+(?=(?:what|which|how|why|when|does|do|at|against)\b)|[?？]\s*(?=\S)|\s*그리고\s*', question, flags=re.I)
    if len(pieces)==1:
        pieces = re.split(r'\s+and\s+|\s*(?:및|이며|이고)\s*',question,flags=re.I)
    pieces = [p.strip(' ,?？') for p in pieces if len(p.strip(' ,?？'))>=6]
    if not pieces:
        pieces=[question]
    # Prefix a compound continuation with its subject clause, not with a gold fact.
    facets=[]
    for i,piece in enumerate(pieces[:3]):
        query=piece if i==0 else pieces[0]+'; '+piece
        facets.append({'facet_id':f'request-{i+1}','request':piece,'query':query,
                       'origin':'question_text','semantic_support_verified':False})
    return facets


def rank_facet_sources(question,sentences,selector_model,minimum_score=-3.0):
    facets=question_facets(question)
    pairs=[(facet['query'],s['text']) for facet in facets for s in sentences]
    values=selector_model.predict(pairs,batch_size=16,show_progress_bar=False,convert_to_numpy=True).reshape(-1).tolist()
    if len(values)!=len(pairs):
        raise RuntimeError('Facet scorer returned an unexpected number of scores')
    representatives=[];members={s['sentence_id']:[] for s in sentences}
    for i,facet in enumerate(facets):
        scores=values[i*len(sentences):(i+1)*len(sentences)]
        best=max(range(len(scores)),key=lambda j:(scores[j]+0.5*sentences[j].get('intent_bonus',0),-j))
        facet['best_relevance_score']=round(float(scores[best]),6)
        facet['candidate_sentence_id']=sentences[best]['sentence_id']
        facet['candidate_above_threshold']=scores[best]>=minimum_score
        if facet['candidate_above_threshold']:
            representatives.append(sentences[best]['sentence_id'])
        for j,score in enumerate(scores):
            if score>=minimum_score and score>=scores[best]-0.8:
                members[sentences[j]['sentence_id']].append(facet['facet_id'])
    return facets,list(dict.fromkeys(representatives)),members


def evidence_plan(question,selected,enabled):
    facets=question_facets(question)
    for facet in facets:
        matches=[s for s in selected if facet['facet_id'] in s.get('requested_facet_ids',[])]
        facet['selected_sentence_ids']=[s['sentence_id'] for s in matches]
        facet['selected_evidence_ids']=list(dict.fromkeys(s['evidence_id'] for s in matches))
        facet['status']='candidate_selected' if matches else 'not_confirmed'
    return {'enabled':bool(enabled),'facets':facets,
            'missing_facet_ids':[f['facet_id'] for f in facets if not f['selected_sentence_ids']],
            'coverage_kind':'cross-encoder relevance and explicit request decomposition; not entailment',
            'gold_used':False}


def align_answer_sentences(selected,translated=None,language=None):
    """Keep an exact source-to-answer correspondence; do not claim semantic proof."""
    if translated is not None and len(translated)!=len(selected):
        raise ValueError('Cannot align a different number of translations and source sentences')
    language=language or ('ko' if translated is not None else 'en')
    return [{'sentence_id':s['sentence_id'],'evidence_id':s['evidence_id'],
             'page':s['page'],'section':s.get('section'),'source_text':s['text'],
             'answer_text':translated[i] if translated is not None else s['text'],
             'answer_language':language,
             'translation_status':'structural_checks_passed' if language=='ko' else 'original_source',
             'requested_facet_ids':s.get('requested_facet_ids',[])} for i,s in enumerate(selected)]
