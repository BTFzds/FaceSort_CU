"""标签格式单元测试。"""

from facesort.tags import apply_folder_person_tags, folder_person_tag


def test_folder_person_tag() -> None:
    assert folder_person_tag(r"E:\photos\groups\a.jpg", "PersonA") == "groups_PersonA"


def test_apply_folder_person_tags() -> None:
    matches = [{"path": r"D:\album\case_identity\x.jpg", "photo_id": 1, "similarity": 0.9}]
    out = apply_folder_person_tags(matches, "PersonObama")
    assert out[0]["person_name"] == "case_identity_PersonObama"
    assert out[0]["source_folder"] == "case_identity"
