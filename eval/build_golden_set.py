"""Build eval/golden_set.jsonl from the question list below.

Each answerable question stores EVIDENCE SNIPPETS (short verbatim phrases from your
documents) instead of hand-typed chunk IDs. This script finds which chunks in your
frozen chunks file contain each snippet, so labels stay correct even when a passage
sits in the overlap between two chunks or is repeated elsewhere in the corpus.

How the evidence is structured
    "evidence": [ group, group, ... ]      each group is a list of alternative snippets
    - A chunk matches a group if it contains ANY snippet in that group.
    - To answer the question fully you need ONE chunk from EACH group.
    The output stores this as "answer_groups" (a list of chunk-ID lists), plus a flat
    "relevant_chunks" list (the union of all groups).

Run from the rag_doc root:
    python eval/build_golden_set.py
    python eval/build_golden_set.py --chunks data/processed/chunks.frozen.jsonl
"""
import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from datetime import date
from pathlib import Path

TYPES = {"lookup", "multi_chunk", "paraphrase", "keyword", "unanswerable"}

QUESTIONS = [
    # ------------------------------------------------------------------ lookup
    {
        "id": "q01", "type": "lookup",
        "question": "What are the three levels of data abstraction in a DBMS?",
        "answer": "Physical (internal) level: how the data is stored. Logical (conceptual) level: the tables, attributes and relationships. View (external) level: the part of the database a user sees.",
        "evidence": [[
            "There are mainly 3 levels of data abstraction",
            "Physical or Internal Level Logical or Conceptual Level View or External Level",
        ]],
    },
    {
        "id": "q02", "type": "lookup",
        "question": "Which symbol is used to draw a weak entity in an ER diagram?",
        "answer": "A double rectangle.",
        "evidence": [[
            "The weak entity is represented by a double rectangle",
            "A weak entity type is represented by a Double Rectangle",
            "Double Rectangle represents a Weak Entity",
            "Double Rectangles: Weak Entity Sets",
        ]],
    },
    {
        "id": "q03", "type": "lookup",
        "question": "In a 3-tier DBMS architecture, what sits between the client and the database server?",
        "answer": "An application server (a middle layer). The client talks to the application server, which communicates with the database system, instead of the client contacting the server directly.",
        "evidence": [[
            "it interacts with an application server which further communicates with the database system",
        ]],
    },
    {
        "id": "q04", "type": "lookup",
        "question": "Which transaction isolation level is the lowest, and what does it allow?",
        "answer": "Read Uncommitted. A transaction may read changes that other transactions have not committed yet, so dirty reads are allowed.",
        "evidence": [[
            "Read Uncommitted is the lowest isolation level",
            "Read Uncommitted: This is the lowest level of isolation",
        ]],
    },
    {
        "id": "q05", "type": "lookup",
        "question": "Which ACID property is known as the 'all or nothing rule', and what does it mean?",
        "answer": "Atomicity: either the entire transaction takes place at once or it does not happen at all; there is no partial execution.",
        "evidence": [[
            "Atomicity is also known as the All or nothing rule",
            "Either all the operations within the transaction are completed successfully, or none of them are",
        ]],
    },
    {
        "id": "q06", "type": "lookup",
        "question": "In log-based recovery, what does the write_item(T, X, old_value, new_value) log entry record?",
        "answer": "That transaction T changed the value of database item X from old_value (the before image) to new_value (the after image).",
        "evidence": [[
            "This log entry records that transaction T changes the value of the database item X",
        ]],
    },
    {
        "id": "q07", "type": "lookup",
        "question": "What is the time complexity of search, insert and delete in a B-Tree?",
        "answer": "O(log n) for all three operations.",
        "evidence": [[
            "the time complexity for operations such as insertion, deletion, and searching is always O(log n)",
            "B-Trees have a guaranteed time complexity of O(log n) for basic operations",
            "the time complexity to search, insert, and delete is O(log n)",
        ]],
    },
    {
        "id": "q08", "type": "lookup",
        "question": "How many clustered indexes can a single table have?",
        "answer": "Only one. A table can have one clustered index but one or more non-clustered indexes.",
        "evidence": [[
            "We can create only one clustered index in a table",
            "there can be only one clustered index or one or more than one non_clustered index",
        ]],
    },
    # ------------------------------------------------------------- multi_chunk
    {
        "id": "q09", "type": "multi_chunk",
        "question": "What are insertion, deletion and update anomalies, and how can they be avoided?",
        "answer": "Insertion anomaly: data cannot be inserted because required fields (for example a primary key value) are missing or the data is incomplete. Deletion anomaly: deleting a record can unintentionally lose related data (for example deleting a customer also deletes their orders). Update anomaly: changing data in one record but not in all related records leaves inconsistencies. They can be avoided or minimized by designing databases that follow the principles of normalization.",
        "evidence": [
            ["Insertion Anomalies: These anomalies occur when it is not possible to insert data into a database"],
            ["Deletion anomalies: These anomalies occur when deleting a record from a database"],
            ["Update anomalies: These anomalies occur when modifying data in a database"],
            ["These anomalies can be avoided or minimized by designing databases that adhere to"],
        ],
    },
    {
        "id": "q10", "type": "multi_chunk",
        "question": "What is attribute closure, and how is it used to find super keys and candidate keys?",
        "answer": "The attribute closure of an attribute set is the set of attributes that can be functionally determined from it. If the closure contains all attributes of the relation, the set is a super key. If in addition no subset of it can determine all attributes, it is a candidate key.",
        "evidence": [
            ["Attribute closure of an attribute set can be defined as set of attributes which can"],
            ["If attribute closure of an attribute set contains all attributes of relation"],
            ["If no subset of this attribute set can functionally determine all attributes of the relation"],
        ],
    },
    {
        "id": "q11", "type": "multi_chunk",
        "question": "According to Codd's rules, how must NULL values be treated, and what does the physical data independence rule require?",
        "answer": "Rule 3 (systematic treatment of NULL values): a NULL must only represent missing, unknown or not applicable values. Rule 8 (physical data independence): changing the physical location of a table must not require changes at the application level.",
        "evidence": [
            ["NULL value in the database must only correspond to missing, unknown, or not applicable values"],
            ["Any modification in the physical location of a table should not enforce modification"],
        ],
    },
    {
        "id": "q12", "type": "multi_chunk",
        "question": "Which join types does SQL support, and which one returns all rows from both tables with NULLs where there is no match?",
        "answer": "SQL supports INNER JOIN, LEFT JOIN, RIGHT JOIN, FULL JOIN and NATURAL JOIN. FULL JOIN combines the results of LEFT JOIN and RIGHT JOIN, so it returns all rows from both tables, with NULL values where there is no match.",
        "evidence": [
            ["Different types of Joins are as follows INNER JOIN LEFT JOIN RIGHT JOIN FULL JOIN NATURAL JOIN"],
            ["FULL JOIN creates the result-set by combining results of both LEFT JOIN and RIGHT JOIN",
             "Full Outer Joins gives all tuples of S and all tuples of R in the result set"],
        ],
    },
    {
        "id": "q13", "type": "multi_chunk",
        "question": "How do the deferred update and immediate update recovery techniques differ, and which one needs both undo and redo?",
        "answer": "Deferred update does not physically update the database on disk until the transaction reaches its commit point (updates are kept in a local workspace), so no undo is needed but redo may be: the no-undo/redo algorithm. Immediate update may change the database before the commit point, but only after the operations are logged, so both undo and redo are needed: the undo/redo algorithm.",
        "evidence": [
            ["Deferred Update: This technique does not physically update the database on disk",
             "If the transaction does not modify the database until it has partially committed"],
            ["Immediate Update: In the immediate update, the database may be updated by some operations",
             "If database modification occur while the transaction is still active"],
            ["hence we require both undo and redo"],
        ],
    },
    {
        "id": "q14", "type": "multi_chunk",
        "question": "What are the main challenges of database security in a DBMS, and which features help enhance it?",
        "answer": "Challenges: authentication and authorization, encryption (including key management), access control, auditing and logging, database design weaknesses such as SQL injection, malicious attacks (hacking, malware, phishing) and physical security. Features that help: backup and recovery, access controls based on roles or permissions, database auditing and testing tools, and data masking.",
        "evidence": [
            ["The challenges of database security in DBMS include"],
            ["Features that are used to enhance database security"],
        ],
    },
    # --------------------------------------------------------------- paraphrase
    {
        "id": "q15", "type": "paraphrase",
        "question": "In a university that keeps separate files per department, what goes wrong when a student changes their phone number?",
        "answer": "The change has to be made in every section that keeps a copy (data redundancy), and if some copies are missed the sections end up with different values for the same data (inconsistency).",
        "evidence": [
            ["he or she has to get it updated in various sections"],
            ["If the Phone number is different in Accounts Section and Academics Section, it will be inconsistent"],
        ],
    },
    {
        "id": "q16", "type": "paraphrase",
        "question": "What problem occurs when two transactions work on the same item and one transaction's write wipes out the other's update?",
        "answer": "The lost update problem: when two or more transactions modify the same data, one transaction's update is overwritten (lost) by another transaction.",
        "evidence": [[
            "Lost update problem occurs when two or more transactions modify the same data",
            "the update being overwritten or lost by another transaction",
        ]],
    },
    {
        "id": "q17", "type": "paraphrase",
        "question": "When splitting one table into two smaller ones, how can we make sure joining them back gives exactly the original data?",
        "answer": "Use a lossless-join decomposition: a natural join of the two new relations returns the original relation. This holds when the attributes of the two relations together cover the original and their common attribute is a candidate key or super key in at least one of them.",
        "evidence": [
            ["if we perform a natural join of relation R1 and R2, it will return the original relation R",
             "it becomes feasible to reconstruct the relation R from decomposed tables R1 and R2 by using Joins"],
            ["the common attribute must be a candidate key or super key in either relation R1, R2, or both",
             "The common attribute must be a key for at least one relation"],
        ],
    },
    {
        "id": "q18", "type": "paraphrase",
        "question": "How can a database find rows without examining every row of a huge table?",
        "answer": "By using an index: it gives direct, fast access to rows through pointers and reduces disk I/O, whereas without an index the server has to scan the whole table.",
        "evidence": [[
            "An index provides direct and fast access to rows in a table",
            "Without indexing, the SQL server has to scan the whole table for your data",
            "If you do not have an index on the column, then a full table scan occurs",
            "Indexing improves database performance by minimizing the number of disc visits",
        ]],
    },
    {
        "id": "q19", "type": "paraphrase",
        "question": "What is it called when a transaction reads data written by another transaction that later fails and is rolled back?",
        "answer": "A dirty read: a transaction reads data that another transaction has not committed (and that may later be rolled back), which creates inconsistency in the database.",
        "evidence": [[
            "A Dirty read is a situation when a transaction reads data that has not yet been committed",
            "Dirty read problem occurs when one transaction updates an item",
        ]],
    },
    {
        "id": "q20", "type": "paraphrase",
        "question": "What kind of attribute is Age when it can be worked out from the date of birth?",
        "answer": "A derived attribute (drawn as a dashed oval in an ER diagram), because its value can be derived from another attribute, for example age from date of birth.",
        "evidence": [[
            "Person age is a derived attribute as it changes over time",
            "An attribute that can be derived from other attributes of the entity type",
            "Age (can be derived from DOB)",
        ]],
    },
    # ----------------------------------------------------------------- keyword
    {
        "id": "q21", "type": "keyword",
        "question": "REVOKE SELECT, UPDATE ON MY_TABLE FROM USER1, USER2",
        "answer": "REVOKE withdraws access privileges previously given with GRANT. Example: REVOKE SELECT, UPDATE ON MY_TABLE FROM USER1, USER2;",
        "evidence": [[
            "REVOKE SELECT, UPDATE ON MY_TABLE FROM USER1, USER2",
            "This command withdraws the user's access privileges given by using the GRANT command",
        ]],
    },
    {
        "id": "q22", "type": "keyword",
        "question": "CREATE SEQUENCE dept_deptid_seq INCREMENT BY 10 START WITH 120",
        "answer": "It creates a sequence that generates unique integers starting at 120 and increasing by 10, up to a maximum of 9999, with no caching and no cycling.",
        "evidence": [[
            "CREATE SEQUENCE dept_deptid_seq INCREMENT BY 10 START WITH 120 MAXVALUE 9999 NOCACHE NOCYCLE",
        ]],
    },
    {
        "id": "q23", "type": "keyword",
        "question": "W_TS(X) R_TS(X)",
        "answer": "W_TS(X) is the largest timestamp of any transaction that executed write(X) successfully; R_TS(X) is the largest timestamp of any transaction that executed read(X) successfully.",
        "evidence": [
            ["W_TS(X) is the largest timestamp of any transaction that executed write(X) successfully",
             "W-timestamp(X): This means the latest time when the data item X has been written into"],
            ["R_TS(X) is the largest timestamp of any transaction that executed read(X) successfully",
             "R-timestamp(X): This means the latest time when the data item X has been read from"],
        ],
    },
    {
        "id": "q24", "type": "keyword",
        "question": "SELECT DISTINCT ADDRESS FROM STUDENT",
        "answer": "DISTINCT returns each different value only once (DELHI, GURGAON, ROHTAK). Without DISTINCT, DELHI would be repeated twice in the result.",
        "evidence": [[
            "If DISTINCT is not used, DELHI will be repeated twice in result set",
            "If we want to retrieve distinct values of an attribute or group of attributes, DISTINCT is used",
        ]],
    },
    {
        "id": "q25", "type": "keyword",
        "question": "Fifth Normal Form Projected Normal Form",
        "answer": "Fifth Normal Form (5NF) is also called Projected Normal Form. The relation must be in Fourth Normal Form and must not be further decomposable without loss.",
        "evidence": [[
            "Fifth Normal Form is also called as Projected Normal Form",
        ]],
    },
    # ------------------------------------------------------------ unanswerable
    {
        "id": "q26", "type": "unanswerable",
        "question": "How does sharding work in MongoDB?",
        "absent_terms": ["sharding", "shard"],
    },
    {
        "id": "q27", "type": "unanswerable",
        "question": "What is the CAP theorem and how does it apply to NoSQL databases?",
        "absent_terms": ["CAP theorem", "partition tolerance"],
    },
    {
        "id": "q28", "type": "unanswerable",
        "question": "How do I tune the InnoDB buffer pool size in MySQL?",
        "absent_terms": ["InnoDB", "buffer pool"],
    },
    {
        "id": "q29", "type": "unanswerable",
        "question": "What is a bitmap index and when should it be used?",
        "absent_terms": ["bitmap"],
    },
    {
        "id": "q30", "type": "unanswerable",
        "question": "How does gradient descent update the weights of a neural network?",
        "absent_terms": ["gradient descent"],
    },
]


# --------------------------------------------------------------------- helpers
def norm(text):
    """Lowercase and keep only letters and digits, so line breaks, bullets,
    quotes, dashes and spacing differences never break a match."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


def load_chunks(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_index(chunks):
    return {c["chunk_id"]: norm(c["text"]) for c in chunks}


def find_chunks(snippet, index):
    key = norm(snippet)
    return [cid for cid, text in index.items() if key in text]


def find_word(term, chunks):
    """Whole-word search (case-insensitive) for the unanswerable-term check.
    Unlike find_chunks, spaces matter here, so 'shard' will not match 'such as hardware'."""
    pattern = r"(?<![a-z0-9])" + r"\s+".join(re.escape(w) for w in term.lower().split()) + r"(?![a-z0-9])"
    return [c["chunk_id"] for c in chunks if re.search(pattern, c["text"].lower())]


def resolve_groups(groups, index):
    """Return (matched, errors, warnings). matched[i] = sorted chunk IDs for group i."""
    matched, errors, warnings = [], [], []
    for g_num, group in enumerate(groups, start=1):
        ids = set()
        missing = []
        for snippet in group:
            hits = find_chunks(snippet, index)
            if hits:
                ids.update(hits)
            else:
                missing.append(snippet)
        if not ids:
            errors.append(f"group {g_num} matched no chunk (first snippet: {group[0]!r})")
        else:
            for snippet in missing:
                warnings.append(f"group {g_num}: snippet not found: {snippet!r}")
        matched.append(sorted(ids))
    return matched, errors, warnings


def single_chunk_cover(matched_groups):
    """Chunks that contain an answer to EVERY group on their own."""
    sets = [set(g) for g in matched_groups]
    return sorted(set.intersection(*sets)) if sets else []


def sha256_of(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chunks", default=None,
                    help="chunks file (default: data/processed/chunks.frozen.jsonl if it exists, "
                         "otherwise data/processed/chunks.jsonl)")
    ap.add_argument("--out", default="eval/golden_set.jsonl")
    ap.add_argument("--only", default=None,
                    help="comma-separated question ids to build, e.g. q09,q10 (for debugging)")
    args = ap.parse_args()

    if args.chunks:
        chunks_path = Path(args.chunks)
    else:
        frozen = Path("data/processed/chunks.frozen.jsonl")
        chunks_path = frozen if frozen.exists() else Path("data/processed/chunks.jsonl")
    if not chunks_path.exists():
        sys.exit(f"Chunks file not found: {chunks_path}")

    chunks = load_chunks(chunks_path)
    index = build_index(chunks)
    print(f"Chunks file: {chunks_path} ({len(chunks)} chunks)")

    questions = QUESTIONS
    if args.only:
        wanted = {q.strip() for q in args.only.split(",")}
        questions = [q for q in QUESTIONS if q["id"] in wanted]

    errors, warnings, entries = [], [], []
    seen_ids = set()

    for q in questions:
        qid = q["id"]
        if qid in seen_ids:
            errors.append(f"{qid}: duplicate question id")
        seen_ids.add(qid)
        if q["type"] not in TYPES:
            errors.append(f"{qid}: unknown type {q['type']!r}")

        if q["type"] == "unanswerable":
            for term in q["absent_terms"]:
                hits = find_word(term, chunks)
                if hits:
                    errors.append(f"{qid}: term {term!r} IS in the corpus ({hits[:3]}); "
                                  "this question is not unanswerable")
            entries.append({
                "id": qid, "type": q["type"], "question": q["question"],
                "answerable": False, "relevant_chunks": [], "answer_groups": [],
                "reference_answer": None,
            })
            continue

        matched, errs, warns = resolve_groups(q["evidence"], index)
        errors += [f"{qid}: {e}" for e in errs]
        warnings += [f"{qid}: {w}" for w in warns]
        if errs:
            continue

        if q["type"] == "multi_chunk":
            cover = single_chunk_cover(matched)
            if cover:
                warnings.append(f"{qid}: marked multi_chunk, but {cover} alone answers every part; "
                                "consider changing its type or evidence")
        union = sorted({cid for g in matched for cid in g})
        if len(union) > 8:
            warnings.append(f"{qid}: {len(union)} relevant chunks (many repeats of this passage)")

        entries.append({
            "id": qid, "type": q["type"], "question": q["question"],
            "answerable": True, "relevant_chunks": union, "answer_groups": matched,
            "reference_answer": q["answer"],
        })

    # ---- report
    print(f"\n{'id':4}  {'type':12} {'groups':>6} {'chunks':>6}  chunk ids")
    for e in entries:
        ids = e["relevant_chunks"]
        shown = ", ".join(ids[:4]) + (" ..." if len(ids) > 4 else "")
        print(f"{e['id']:4}  {e['type']:12} {len(e['answer_groups']):>6} {len(ids):>6}  {shown}")

    if warnings:
        print("\nWARNINGS")
        for w in warnings:
            print("  " + w)
    if errors:
        print("\nERRORS (nothing was written)")
        for e in errors:
            print("  " + e)
        sys.exit(1)

    # ---- write
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    meta = {
        "chunks_file": str(chunks_path),
        "chunks_sha256": sha256_of(chunks_path),
        "n_chunks": len(chunks),
        "n_questions": len(entries),
        "by_type": dict(Counter(e["type"] for e in entries)),
        "built": date.today().isoformat(),
    }
    meta_path = out.with_name(out.stem + ".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"\nWrote {len(entries)} questions to {out}")
    print(f"Wrote {meta_path} (chunks sha256 {meta['chunks_sha256'][:12]}...)")


if __name__ == "__main__":
    main()
