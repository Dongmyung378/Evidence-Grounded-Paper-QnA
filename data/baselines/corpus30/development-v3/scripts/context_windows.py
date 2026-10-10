"""Restore same-page section context across overlapping production chunks."""
import hashlib


def _hash(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def merge_overlapping_texts(parts):
    result=parts[0] if parts else ''
    for text in parts[1:]:
        overlap=0
        for size in range(min(len(result),len(text),400),19,-1):
            if result.endswith(text[:size]):
                overlap=size;break
        result += text[overlap:] if overlap else '\n'+text
    return result


def expand_context_windows(evidence,chunks,paper_id,max_chars=3200):
    lookup={c['chunk_id']:c for c in chunks if c['paper_id']==paper_id}
    groups={}
    for chunk in lookup.values():
        groups.setdefault((chunk['page'],chunk.get('section_index'),chunk.get('section')),[]).append(chunk)
    for group in groups.values():group.sort(key=lambda c:(c.get('chunk_index',0),c['chunk_id']))
    expanded=[];origins=[]
    for item in evidence:
        if item['paper_id']!=paper_id:
            raise ValueError('Context window evidence escaped the selected paper')
        original=lookup.get(item['chunk_id'])
        if not original or original['text']!=item['text']:
            expanded.append(item);continue
        group=groups[(original['page'],original.get('section_index'),original.get('section'))]
        index=next(i for i,c in enumerate(group) if c['chunk_id']==item['chunk_id'])
        chosen=[original]
        for neighbor in (index-1,index+1):
            if 0<=neighbor<len(group):
                candidate=sorted(chosen+[group[neighbor]],key=lambda c:(c.get('chunk_index',0),c['chunk_id']))
                if len(merge_overlapping_texts([c['text'] for c in candidate]))<=max_chars:
                    chosen=candidate
        text=merge_overlapping_texts([c['text'] for c in chosen])
        if len(chosen)==1:
            expanded.append(item);continue
        origin={'evidence_id':item['evidence_id'],'paper_id':paper_id,'page':item['page'],
                'primary_chunk_id':item['chunk_id'],'source_chunk_ids':[c['chunk_id'] for c in chosen],
                'source_sha256':{c['chunk_id']:_hash(c['text']) for c in chosen},'window_sha256':_hash(text),
                'scope':'same source page and section; at most one preceding and one following chunk'}
        expanded.append({**item,'text':text,'context_window':origin,
            'locator':{**item['locator'],'source_chunk_ids':origin['source_chunk_ids']}})
        origins.append(origin)
    return expanded,origins
