
import numpy as np

TWO_PERSON_IDX = [
    49,50,51,52,53,54,55,56,57,58,59,
    105,106,107,108,109,110,111,112,113,114,115,116,117,118,119
]

def get_topk_acc(cls_score, action_label, 
                 top_k=1, 
                 mode='overall', 
                 subset=None,
                 two_person_set=TWO_PERSON_IDX):
    
    # 1) CPU numpy 변환
    scores_np = cls_score.detach().cpu().numpy()       # (B, num_classes)
    labels_np = action_label.detach().cpu().numpy()    # (B,)

    # 2) Top-K 예측 여부 판단
    if top_k == 1:
        preds = np.argmax(scores_np, axis=1)  # (B,)
        correct_mask = (preds == labels_np)
    else:
        # np.argsort -> 오름차순 정렬 인덱스
        sorted_idx = np.argsort(scores_np, axis=1)  # (B, num_classes)
        topk_idx = sorted_idx[:, -top_k:]           # (B, top_k) - 가장 높은 점수 K개
        correct_mask = np.array([
            (labels_np[i] in topk_idx[i]) 
            for i in range(len(labels_np))
        ])

    # 3) Overall accuracy
    acc_overall = correct_mask.mean() * 100.0

    # 모드별로 분기
    if mode == 'overall':
        # 전체 정확도만 반환
        return {'acc_overall': acc_overall}

    elif mode == 'split':
        # single vs two-person
        two_mask = np.array([lbl in two_person_set for lbl in labels_np])
        single_mask = ~two_mask
        
        if single_mask.sum() > 0:
            acc_single = correct_mask[single_mask].mean() * 100.0
        else:
            acc_single = 0.0
        
        if two_mask.sum() > 0:
            acc_two = correct_mask[two_mask].mean() * 100.0
        else:
            acc_two = 0.0
        
        return {
            'acc_overall': acc_overall,
            'acc_single': acc_single,
            'acc_two': acc_two
        }
    
    elif mode == 'subset':
        """
        subset에 예: np.array([1,3,5])가 들어오면
        해당 label인 샘플만 골라서 accuracy 계산.
        - acc_subset: subset 전체에 대한 정확도
        - acc_subset_each: 각 클래스별로 정확도
        """
        if subset is None or len(subset) == 0:
            raise ValueError("mode='subset'인 경우, subset을 지정해야 합니다.")
        
        # subset 전체 마스크
        subset_mask = np.isin(labels_np, subset)
        if subset_mask.sum() > 0:
            acc_subset = correct_mask[subset_mask].mean() * 100.0
        else:
            acc_subset = 0.0
        
        # 각 클래스별 정확도
        acc_subset_each = {}
        for cls_id in subset:
            cls_mask = (labels_np == cls_id)
            if cls_mask.sum() > 0:
                acc_subset_each[cls_id] = correct_mask[cls_mask].mean() * 100.0
            else:
                acc_subset_each[cls_id] = 0.0
        
        return {
            'acc_overall': acc_overall,
            'acc_subset': acc_subset,
            'acc_subset_each': acc_subset_each
        }

    else:
        raise ValueError(f"Unknown mode: {mode}. Must be 'overall','split','subset'.")


def get_top1_acc(cls_score, action_label, mode='overall', subset=None):
    """
    Top-1 정확도 계산
    mode='overall','split','subset' 중 선택 가능
    subset: mode='subset'일 때 관심 클래스
    """
    return get_topk_acc(cls_score, action_label, 
                        top_k=1, 
                        mode=mode, 
                        subset=subset, 
                        two_person_set=TWO_PERSON_IDX)

def get_top5_acc(cls_score, action_label, mode='overall', subset=None):
    """
    Top-5 정확도 계산
    mode='overall','split','subset' 중 선택 가능
    subset: mode='subset'일 때 관심 클래스
    """
    return get_topk_acc(cls_score, action_label, 
                        top_k=5, 
                        mode=mode, 
                        subset=subset, 
                        two_person_set=TWO_PERSON_IDX)
