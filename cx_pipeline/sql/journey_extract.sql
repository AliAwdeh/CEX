WITH params AS (
    SELECT
        /* daily default: resolver = today, append = today going back 1 day.
           Upper bounds are exclusive, so DATEADD(day, 1, CURRENT_DATE)
           includes all of today. */
        CURRENT_DATE                    AS RESOLVER_WINDOW_START,
        DATEADD(day,  1, CURRENT_DATE)  AS RESOLVER_WINDOW_END,

        DATEADD(day, -1, CURRENT_DATE)  AS APPEND_WINDOW_START,
        DATEADD(day,  1, CURRENT_DATE)  AS APPEND_WINDOW_END,

        /* 1 = keep journeys whose entire window is a single broadcast. */
        1::NUMBER                       AS MIN_APPENDED_MESSAGES
),

date_window AS (
    SELECT
        RESOLVER_WINDOW_START,
        RESOLVER_WINDOW_END,
        APPEND_WINDOW_START,
        APPEND_WINDOW_END,
        MIN_APPENDED_MESSAGES
    FROM params
),

/* ---------------------------------------------------------------------------
   Pruned to the widest window either gate can reference. Skills live on the
   conversation row itself, so date-bounding here loses nothing and keeps the
   query off a full scan of CC_CONVERSATIONS.
   ------------------------------------------------------------------------ */
conversation_base AS (
    SELECT
        c.*,

        UPPER(TRIM(COALESCE(c.INITIAL_SKILL, '')))
            AS INITIAL_SKILL_UPPER,

        UPPER(TRIM(COALESCE(c.LAST_SKILL, '')))
            AS LAST_SKILL_UPPER,

        UPPER(TRIM(COALESCE(c.JOINED_SKILLS, '')))
            AS JOINED_SKILLS_UPPER

    FROM BA_VIEWS.CHATCC_SILVER.CC_CONVERSATIONS c

    CROSS JOIN date_window dw

    WHERE
        c.START_DATE >= LEAST(
            dw.RESOLVER_WINDOW_START,
            dw.APPEND_WINDOW_START
        )

        AND c.START_DATE < GREATEST(
            dw.RESOLVER_WINDOW_END,
            dw.APPEND_WINDOW_END
        )
),

conversation_skill_tokens AS (
    SELECT
        c.CONVERSATION_ID,
        c.INITIAL_SKILL_UPPER AS SKILL_NAME

    FROM conversation_base c

    WHERE NULLIF(c.INITIAL_SKILL_UPPER, '') IS NOT NULL

    UNION ALL

    SELECT
        c.CONVERSATION_ID,
        c.LAST_SKILL_UPPER AS SKILL_NAME

    FROM conversation_base c

    WHERE NULLIF(c.LAST_SKILL_UPPER, '') IS NOT NULL

    UNION ALL

    SELECT
        c.CONVERSATION_ID,
        UPPER(TRIM(f.VALUE::STRING)) AS SKILL_NAME

    FROM conversation_base c,
         LATERAL FLATTEN(
             INPUT => SPLIT(
                 COALESCE(c.JOINED_SKILLS, ''),
                 ','
             )
         ) f

    WHERE NULLIF(TRIM(f.VALUE::STRING), '') IS NOT NULL
),

distinct_conversation_skills AS (
    SELECT DISTINCT
        CONVERSATION_ID,
        SKILL_NAME

    FROM conversation_skill_tokens

    WHERE NULLIF(TRIM(SKILL_NAME), '') IS NOT NULL
),

conversation_skill_validation AS (
    SELECT
        c.CONVERSATION_ID,

        COUNT(dcs.SKILL_NAME)
            AS DISTINCT_SKILL_COUNT,

        COUNT_IF(
            dcs.SKILL_NAME IN (
                'GPT_RESOLVERS_BOT',
                'GPT_MV_RESOLVERS',
                'GPT_MV_RESOLVERS_QUEUE',
                'MV_RESOLVERS_SENIORS',
                'MV_CALLERS',
                'GPT_MV_CALLERS_QUEUE'
            )
        ) AS ALLOWED_SKILL_COUNT,

        COUNT_IF(
            dcs.SKILL_NAME NOT IN (
                'GPT_RESOLVERS_BOT',
                'GPT_MV_RESOLVERS',
                'GPT_MV_RESOLVERS_QUEUE',
                'MV_RESOLVERS_SENIORS',
                'MV_CALLERS',
                'GPT_MV_CALLERS_QUEUE'
            )
        ) AS UNAUTHORIZED_SKILL_COUNT,

        COUNT_IF(
            dcs.SKILL_NAME = 'GPT_RESOLVERS_BOT'
        ) > 0 AS HAS_GPT_RESOLVERS_BOT,

        COUNT_IF(
            dcs.SKILL_NAME = 'GPT_MV_RESOLVERS'
        ) > 0 AS HAS_GPT_MV_RESOLVERS,

        COUNT_IF(
            dcs.SKILL_NAME = 'GPT_MV_RESOLVERS_QUEUE'
        ) > 0 AS HAS_GPT_MV_RESOLVERS_QUEUE,

        COUNT_IF(
            dcs.SKILL_NAME = 'MV_RESOLVERS_SENIORS'
        ) > 0 AS HAS_MV_RESOLVERS_SENIORS,

        COUNT_IF(
            dcs.SKILL_NAME = 'MV_CALLERS'
        ) > 0 AS HAS_MV_CALLERS,

        COUNT_IF(
            dcs.SKILL_NAME = 'GPT_MV_CALLERS_QUEUE'
        ) > 0 AS HAS_GPT_MV_CALLERS_QUEUE,

        LISTAGG(
            DISTINCT IFF(
                dcs.SKILL_NAME NOT IN (
                    'GPT_RESOLVERS_BOT',
                    'GPT_MV_RESOLVERS',
                    'GPT_MV_RESOLVERS_QUEUE',
                    'MV_RESOLVERS_SENIORS',
                    'MV_CALLERS',
                    'GPT_MV_CALLERS_QUEUE'
                ),
                dcs.SKILL_NAME,
                NULL
            ),
            ', '
        ) WITHIN GROUP (
            ORDER BY IFF(
                dcs.SKILL_NAME NOT IN (
                    'GPT_RESOLVERS_BOT',
                    'GPT_MV_RESOLVERS',
                    'GPT_MV_RESOLVERS_QUEUE',
                    'MV_RESOLVERS_SENIORS',
                    'MV_CALLERS',
                    'GPT_MV_CALLERS_QUEUE'
                ),
                dcs.SKILL_NAME,
                NULL
            )
        ) AS UNAUTHORIZED_SKILLS,

        /*
        Skill containment is now the ONLY test.

        Dropped on purpose:
          - the "GPT_RESOLVERS_BOT without GPT_MV_RESOLVERS -> FALSE" rule.
            A conversation that only ever saw the bot, or that carries a single
            broadcast, is a legitimate part of the journey.
          - the "no skill at all -> FALSE" rule. A conversation with no skill
            has no skill OUTSIDE the allowed set, so it stays in scope.

        UNAUTHORIZED_SKILLS / UNAUTHORIZED_SKILL_COUNT remain in the output for
        anyone who wants to re-tighten this downstream.
        */
        CASE
            WHEN COUNT_IF(
                dcs.SKILL_NAME NOT IN (
                    'GPT_RESOLVERS_BOT',
                    'GPT_MV_RESOLVERS',
                    'GPT_MV_RESOLVERS_QUEUE',
                    'MV_RESOLVERS_SENIORS',
                    'MV_CALLERS',
                    'GPT_MV_CALLERS_QUEUE'
                )
            ) > 0
            THEN FALSE

            ELSE TRUE
        END AS IS_ALLOWED_RESOLVER_CONVERSATION

    FROM conversation_base c

    LEFT JOIN distinct_conversation_skills dcs
        ON c.CONVERSATION_ID = dcs.CONVERSATION_ID

    GROUP BY
        c.CONVERSATION_ID
),

/* ---------------------------------------------------------------------------
   Which customers are in scope.

   Anchored on START_DATE only. The previous version also required
   END_DATE IS NOT NULL and END_DATE inside the window, which silently
   excluded every OPEN / INACTIVE conversation and anything still running at
   the window boundary -- i.e. exactly the open tickets this feed exists to
   append to.
   ------------------------------------------------------------------------ */
resolver_customers AS (
    SELECT DISTINCT
        c.CUSTOMER_PHONE

    FROM conversation_base c

    JOIN conversation_skill_validation csv
        ON c.CONVERSATION_ID = csv.CONVERSATION_ID

    CROSS JOIN date_window dw

    WHERE
        NULLIF(TRIM(c.CUSTOMER_PHONE), '') IS NOT NULL

        AND c.START_DATE >= dw.RESOLVER_WINDOW_START
        AND c.START_DATE < dw.RESOLVER_WINDOW_END

        AND c.INITIAL_SKILL_UPPER NOT LIKE '%MV_COLLECT_INFO%'
        AND c.LAST_SKILL_UPPER NOT LIKE '%MV_COLLECT_INFO%'
        AND c.JOINED_SKILLS_UPPER NOT LIKE '%MV_COLLECT_INFO%'

        AND c.INITIAL_SKILL_UPPER IN (
            'GPT_MV_RESOLVERS',
            'GPT_RESOLVERS_BOT'
        )

        AND csv.HAS_GPT_MV_RESOLVERS = TRUE
),

/* ---------------------------------------------------------------------------
   Every conversation those customers had in the append window, skill mix
   included. This is the denominator for the journey reporting columns.
   ------------------------------------------------------------------------ */
journey_conversations AS (
    SELECT
        c.CONVERSATION_ID,
        c.START_DATE,
        c.END_DATE,
        c.STATUS,
        c.INITIAL_SKILL,
        c.LAST_SKILL,
        c.JOINED_SKILLS,

        c.AGENT_FULL_NAME
            AS CONVERSATION_AGENT_FULL_NAME,

        c.AGENT_LOGIN_NAME
            AS CONVERSATION_AGENT_LOGIN_NAME,

        c.CUSTOMER_PHONE,
        c.CUSTOMER_NAME,

        CASE
            WHEN
                c.INITIAL_SKILL_UPPER IN (
                    'GPT_MV_RESOLVERS',
                    'GPT_RESOLVERS_BOT'
                )
                AND csv.HAS_GPT_MV_RESOLVERS = TRUE
            THEN TRUE
            ELSE FALSE
        END AS IS_REQUIRED_RESOLVER_CONVERSATION,

        csv.IS_ALLOWED_RESOLVER_CONVERSATION,
        csv.DISTINCT_SKILL_COUNT,
        csv.ALLOWED_SKILL_COUNT,
        csv.UNAUTHORIZED_SKILL_COUNT,
        csv.UNAUTHORIZED_SKILLS,

        csv.HAS_GPT_RESOLVERS_BOT,
        csv.HAS_GPT_MV_RESOLVERS,
        csv.HAS_GPT_MV_RESOLVERS_QUEUE,
        csv.HAS_MV_RESOLVERS_SENIORS,
        csv.HAS_MV_CALLERS,
        csv.HAS_GPT_MV_CALLERS_QUEUE

    FROM conversation_base c

    JOIN resolver_customers rc
        ON c.CUSTOMER_PHONE = rc.CUSTOMER_PHONE

    JOIN conversation_skill_validation csv
        ON c.CONVERSATION_ID = csv.CONVERSATION_ID

    CROSS JOIN date_window dw

    WHERE
        NULLIF(TRIM(c.CUSTOMER_PHONE), '') IS NOT NULL

        AND c.START_DATE >= dw.APPEND_WINDOW_START
        AND c.START_DATE < dw.APPEND_WINDOW_END

        AND c.INITIAL_SKILL_UPPER NOT LIKE '%MV_COLLECT_INFO%'
        AND c.LAST_SKILL_UPPER NOT LIKE '%MV_COLLECT_INFO%'
        AND c.JOINED_SKILLS_UPPER NOT LIKE '%MV_COLLECT_INFO%'
),

/*
Conversation-level containment replaces the old journey-level 50% ratio gate.
Off-topic conversations drop out one at a time; the customer is never lost.
*/
target_conversations AS (
    SELECT *
    FROM journey_conversations
    WHERE IS_ALLOWED_RESOLVER_CONVERSATION = TRUE
),

/*
Reporting only -- no HAVING. The old
  HAVING other_skill_ratio <= 0.50
removed 43% of customers on an 8-day sample, which contradicts needing the
whole journey present for daily ticket appends.
*/
journey_skill_validation AS (
    SELECT
        CUSTOMER_PHONE,

        COUNT(DISTINCT CONVERSATION_ID)
            AS TOTAL_CONVERSATIONS_IN_JOURNEY,

        COUNT(
            DISTINCT IFF(
                IS_ALLOWED_RESOLVER_CONVERSATION = TRUE,
                CONVERSATION_ID,
                NULL
            )
        ) AS ALLOWED_RESOLVER_CONVERSATIONS,

        COUNT(
            DISTINCT IFF(
                IS_ALLOWED_RESOLVER_CONVERSATION = FALSE,
                CONVERSATION_ID,
                NULL
            )
        ) AS OTHER_SKILL_CONVERSATIONS,

        COUNT(
            DISTINCT IFF(
                UNAUTHORIZED_SKILL_COUNT > 0,
                CONVERSATION_ID,
                NULL
            )
        ) AS CONVERSATIONS_WITH_UNAUTHORIZED_SKILLS,

        COUNT(
            DISTINCT IFF(
                IS_ALLOWED_RESOLVER_CONVERSATION = FALSE,
                CONVERSATION_ID,
                NULL
            )
        )::FLOAT
        / NULLIF(
            COUNT(DISTINCT CONVERSATION_ID),
            0
        ) AS OTHER_SKILL_CONVERSATION_RATIO

    FROM journey_conversations

    GROUP BY
        CUSTOMER_PHONE
),

/*
RAG retrieval is dead upstream: chunks_to_fetch events fell to zero from the
week of 2026-08-10. These two CTEs are kept so the RAG output columns stay in
the contract and refill automatically if retrieval ever returns. They
currently resolve to NULL / 0 / FALSE for every row.
*/
rag_chunk_events AS (
    SELECT
        m.CONVERSATION_ID,

        m.MESSAGE_TIME
            AS CHUNK_TIME,

        TRY_PARSE_JSON(m.TEXT):chunks_to_fetch
            AS CHUNKS_FETCHED,

        TRY_PARSE_JSON(m.TEXT):justification::STRING
            AS CHUNK_JUSTIFICATION

    FROM BA_VIEWS.CHATCC_SILVER.CC_MESSAGES m

    JOIN target_conversations tc
        ON m.CONVERSATION_ID = tc.CONVERSATION_ID

    CROSS JOIN date_window dw

    WHERE
        m.MESSAGE_TIME >= dw.APPEND_WINDOW_START
        AND m.MESSAGE_TIME < dw.APPEND_WINDOW_END

        AND UPPER(TRIM(COALESCE(m.MESSAGE_SKILL, ''))) = 'GPT_MV_RESOLVERS'
        AND UPPER(TRIM(COALESCE(m.SENT_BY, ''))) = 'SYSTEM'

        AND TRY_PARSE_JSON(m.TEXT):chunks_to_fetch IS NOT NULL
),

resolver_bot_messages AS (
    SELECT
        m.CONVERSATION_ID,

        m.MESSAGE_TIME
            AS BOT_MESSAGE_TIME,

        LAG(m.MESSAGE_TIME) OVER (
            PARTITION BY m.CONVERSATION_ID
            ORDER BY
                m.MESSAGE_TIME,
                COALESCE(m.AGENT_FULL_NAME, ''),
                COALESCE(m.TEXT, ''),
                COALESCE(m.VN_CONTENT, '')
        ) AS PREVIOUS_BOT_MESSAGE_TIME

    FROM BA_VIEWS.CHATCC_SILVER.CC_MESSAGES m

    JOIN target_conversations tc
        ON m.CONVERSATION_ID = tc.CONVERSATION_ID

    CROSS JOIN date_window dw

    WHERE
        m.MESSAGE_TIME >= dw.APPEND_WINDOW_START
        AND m.MESSAGE_TIME < dw.APPEND_WINDOW_END

        AND UPPER(TRIM(COALESCE(m.MESSAGE_SKILL, ''))) = 'GPT_MV_RESOLVERS'

        AND UPPER(TRIM(COALESCE(m.SENT_BY, ''))) NOT IN (
            'CONSUMER',
            'SYSTEM'
        )

        AND NULLIF(TRIM(m.TEXT), '') IS NOT NULL

        AND m.TOOL_CALLS IS NULL
        AND COALESCE(m.INTERNAL, FALSE) = FALSE
),

latest_rag_for_bot_message AS (
    SELECT
        bot.CONVERSATION_ID,
        bot.BOT_MESSAGE_TIME,

        chunk.CHUNK_TIME,
        chunk.CHUNKS_FETCHED,
        chunk.CHUNK_JUSTIFICATION

    FROM resolver_bot_messages bot

    LEFT JOIN rag_chunk_events chunk
        ON bot.CONVERSATION_ID = chunk.CONVERSATION_ID

        AND chunk.CHUNK_TIME < bot.BOT_MESSAGE_TIME

        AND (
            bot.PREVIOUS_BOT_MESSAGE_TIME IS NULL
            OR chunk.CHUNK_TIME > bot.PREVIOUS_BOT_MESSAGE_TIME
        )

    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY
            bot.CONVERSATION_ID,
            bot.BOT_MESSAGE_TIME

        ORDER BY
            chunk.CHUNK_TIME DESC NULLS LAST
    ) = 1
),

clean_messages AS (
    SELECT
        tc.CUSTOMER_PHONE,
        tc.CUSTOMER_NAME,

        tc.CONVERSATION_ID,
        tc.START_DATE,
        tc.END_DATE,
        tc.STATUS,
        tc.INITIAL_SKILL,
        tc.LAST_SKILL,
        tc.JOINED_SKILLS,

        tc.CONVERSATION_AGENT_FULL_NAME,
        tc.CONVERSATION_AGENT_LOGIN_NAME,

        tc.IS_REQUIRED_RESOLVER_CONVERSATION,
        tc.IS_ALLOWED_RESOLVER_CONVERSATION,

        tc.DISTINCT_SKILL_COUNT,
        tc.ALLOWED_SKILL_COUNT,
        tc.UNAUTHORIZED_SKILL_COUNT,
        tc.UNAUTHORIZED_SKILLS,

        jsv.TOTAL_CONVERSATIONS_IN_JOURNEY,
        jsv.ALLOWED_RESOLVER_CONVERSATIONS,
        jsv.OTHER_SKILL_CONVERSATIONS,
        jsv.CONVERSATIONS_WITH_UNAUTHORIZED_SKILLS,
        jsv.OTHER_SKILL_CONVERSATION_RATIO,

        m.MESSAGE_TIME,
        m.MESSAGE_SKILL,
        m.SENT_BY,
        m.AGENT_FULL_NAME,
        m.TEXT,
        m.VN_CONTENT,

        rag.CHUNK_TIME,
        rag.CHUNKS_FETCHED,
        rag.CHUNK_JUSTIFICATION,

        IFF(
            rag.CHUNK_TIME IS NOT NULL,
            ARRAY_CONSTRUCT(
                OBJECT_CONSTRUCT_KEEP_NULL(
                    'chunk_time',
                    rag.CHUNK_TIME,
                    'chunks_fetched',
                    rag.CHUNKS_FETCHED,
                    'justification',
                    rag.CHUNK_JUSTIFICATION
                )
            ),
            NULL
        ) AS RAG_RETRIEVALS,

        IFF(
            rag.CHUNK_TIME IS NOT NULL,
            1,
            0
        ) AS RAG_RETRIEVAL_COUNT

    FROM BA_VIEWS.CHATCC_SILVER.CC_MESSAGES m

    JOIN target_conversations tc
        ON m.CONVERSATION_ID = tc.CONVERSATION_ID

    JOIN journey_skill_validation jsv
        ON tc.CUSTOMER_PHONE = jsv.CUSTOMER_PHONE

    LEFT JOIN latest_rag_for_bot_message rag
        ON m.CONVERSATION_ID = rag.CONVERSATION_ID
        AND m.MESSAGE_TIME = rag.BOT_MESSAGE_TIME

        AND UPPER(TRIM(COALESCE(m.MESSAGE_SKILL, ''))) = 'GPT_MV_RESOLVERS'

        AND UPPER(TRIM(COALESCE(m.SENT_BY, ''))) NOT IN (
            'CONSUMER',
            'SYSTEM'
        )

    CROSS JOIN date_window dw

    WHERE
        m.MESSAGE_TIME >= dw.APPEND_WINDOW_START
        AND m.MESSAGE_TIME < dw.APPEND_WINDOW_END

        AND (
            NULLIF(TRIM(m.TEXT), '') IS NOT NULL
            OR NULLIF(TRIM(m.VN_CONTENT), '') IS NOT NULL
        )

        /*
        Always retain rows containing VN_CONTENT.

        Rows without VN_CONTENT must remain normal visible messages:
        no tool calls and not internal.

        SENT_BY = 'SYSTEM' broadcasts pass this test (they are not INTERNAL and
        carry no TOOL_CALLS), which is what keeps broadcast-only conversations
        in the feed.
        */
        AND (
            NULLIF(TRIM(m.VN_CONTENT), '') IS NOT NULL

            OR (
                m.TOOL_CALLS IS NULL
                AND COALESCE(m.INTERNAL, FALSE) = FALSE
            )
        )

        /*
        Exclude the raw RAG retrieval system event. No-op since 2026-08-10;
        retained as a guard.
        */
        AND NOT (
            UPPER(TRIM(COALESCE(m.MESSAGE_SKILL, ''))) = 'GPT_MV_RESOLVERS'

            AND UPPER(TRIM(COALESCE(m.SENT_BY, ''))) = 'SYSTEM'

            AND TRY_PARSE_JSON(m.TEXT):chunks_to_fetch IS NOT NULL
        )
),

deduped_messages AS (
    SELECT
        *

    FROM clean_messages

    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY
            CONVERSATION_ID,
            MESSAGE_TIME,
            COALESCE(MESSAGE_SKILL, ''),
            COALESCE(SENT_BY, ''),
            COALESCE(AGENT_FULL_NAME, ''),
            COALESCE(TEXT, ''),
            COALESCE(VN_CONTENT, '')

        ORDER BY
            MESSAGE_TIME,
            CONVERSATION_ID
    ) = 1
),

structured_messages AS (
    SELECT
        CUSTOMER_PHONE,
        CUSTOMER_NAME,

        ROW_NUMBER() OVER (
            PARTITION BY CUSTOMER_PHONE

            ORDER BY
                MESSAGE_TIME,
                CONVERSATION_ID,
                COALESCE(MESSAGE_SKILL, ''),
                COALESCE(SENT_BY, ''),
                COALESCE(AGENT_FULL_NAME, ''),
                COALESCE(TEXT, ''),
                COALESCE(VN_CONTENT, '')
        ) AS APPENDED_MESSAGE_INDEX,

        CONVERSATION_ID,

        START_DATE
            AS CONVERSATION_START_DATE,

        END_DATE
            AS CONVERSATION_END_DATE,

        STATUS
            AS CONVERSATION_STATUS,

        INITIAL_SKILL,
        LAST_SKILL,
        JOINED_SKILLS,

        IS_REQUIRED_RESOLVER_CONVERSATION,
        IS_ALLOWED_RESOLVER_CONVERSATION,

        DISTINCT_SKILL_COUNT,
        ALLOWED_SKILL_COUNT,
        UNAUTHORIZED_SKILL_COUNT,
        UNAUTHORIZED_SKILLS,

        TOTAL_CONVERSATIONS_IN_JOURNEY,
        ALLOWED_RESOLVER_CONVERSATIONS,
        OTHER_SKILL_CONVERSATIONS,
        CONVERSATIONS_WITH_UNAUTHORIZED_SKILLS,
        OTHER_SKILL_CONVERSATION_RATIO,

        CONVERSATION_AGENT_FULL_NAME,
        CONVERSATION_AGENT_LOGIN_NAME,

        MESSAGE_TIME,
        MESSAGE_SKILL,

        CHUNK_TIME,
        CHUNKS_FETCHED,
        CHUNK_JUSTIFICATION,

        IFF(
            RAG_RETRIEVAL_COUNT = 1,
            TRUE,
            FALSE
        ) AS HAS_RAG_RETRIEVAL,

        RAG_RETRIEVAL_COUNT,
        RAG_RETRIEVALS,

        CASE
            WHEN NULLIF(TRIM(VN_CONTENT), '') IS NOT NULL
                THEN 'customer'

            WHEN UPPER(TRIM(COALESCE(SENT_BY, ''))) = 'CONSUMER'
                THEN 'customer'

            ELSE 'agent'
        END AS SENDER_ROLE,

        /*
        SENT_BY is one of SYSTEM | CONSUMER | BOT | AGENT. RAW_SENDER_ROLE
        carries it through untouched so downstream can separate a broadcast
        (SYSTEM) from the bot (BOT) from a human (AGENT); SENDER_ROLE collapses
        all three non-customer cases to 'agent'.
        */
        COALESCE(
            SENT_BY,
            'UNKNOWN'
        ) AS RAW_SENDER_ROLE,

        AGENT_FULL_NAME
            AS MESSAGE_AGENT_FULL_NAME,

        CASE
            WHEN NULLIF(TRIM(VN_CONTENT), '') IS NOT NULL
            THEN
                CONCAT(
                    IFF(
                        NULLIF(TRIM(TEXT), '') IS NOT NULL,
                        TRIM(TEXT) || '\n\n',
                        ''
                    ),

                    'Note for evaluator: The following text is a transcription of a voice note, document, or image sent by the customer. Use it only as supporting context for evaluation.',
                    '\n',

                    TRIM(VN_CONTENT)
                )

            ELSE
                NULLIF(TRIM(TEXT), '')
        END AS MESSAGE_TEXT

    FROM deduped_messages
),

eligible_customers AS (
    SELECT
        sm.CUSTOMER_PHONE,

        COUNT(*)
            AS TOTAL_VISIBLE_MESSAGES_IN_APPENDED_JOURNEY,

        COUNT(DISTINCT sm.CONVERSATION_ID)
            AS APPENDED_CONVERSATION_COUNT,

        LISTAGG(
            DISTINCT sm.CONVERSATION_ID,
            ', '
        ) WITHIN GROUP (
            ORDER BY sm.CONVERSATION_ID
        ) AS APPENDED_CONVERSATION_IDS,

        COUNT_IF(
            sm.SENDER_ROLE = 'customer'
        ) AS CUSTOMER_MESSAGE_COUNT_IN_APPENDED_JOURNEY,

        COUNT_IF(
            sm.SENDER_ROLE = 'agent'
        ) AS AGENT_MESSAGE_COUNT_IN_APPENDED_JOURNEY,

        COUNT_IF(
            sm.HAS_RAG_RETRIEVAL = TRUE
        ) AS MESSAGES_WITH_RAG_RETRIEVAL,

        SUM(
            sm.RAG_RETRIEVAL_COUNT
        ) AS TOTAL_RAG_RETRIEVAL_EVENTS,

        MAX(sm.TOTAL_CONVERSATIONS_IN_JOURNEY)
            AS TOTAL_CONVERSATIONS_IN_JOURNEY,

        MAX(sm.ALLOWED_RESOLVER_CONVERSATIONS)
            AS ALLOWED_RESOLVER_CONVERSATIONS,

        MAX(sm.OTHER_SKILL_CONVERSATIONS)
            AS OTHER_SKILL_CONVERSATIONS,

        MAX(sm.CONVERSATIONS_WITH_UNAUTHORIZED_SKILLS)
            AS CONVERSATIONS_WITH_UNAUTHORIZED_SKILLS,

        MAX(sm.OTHER_SKILL_CONVERSATION_RATIO)
            AS OTHER_SKILL_CONVERSATION_RATIO

    FROM structured_messages sm

    CROSS JOIN date_window dw

    GROUP BY
        sm.CUSTOMER_PHONE,
        dw.MIN_APPENDED_MESSAGES

    HAVING
        COUNT(*) >= dw.MIN_APPENDED_MESSAGES
),

conversation_counts AS (
    SELECT
        CONVERSATION_ID,

        COUNT(*)
            AS TOTAL_VISIBLE_MESSAGES_IN_CONVERSATION,

        COUNT_IF(
            SENDER_ROLE = 'customer'
        ) AS CUSTOMER_MESSAGE_COUNT_IN_CONVERSATION,

        COUNT_IF(
            SENDER_ROLE = 'agent'
        ) AS AGENT_MESSAGE_COUNT_IN_CONVERSATION,

        COUNT_IF(
            HAS_RAG_RETRIEVAL = TRUE
        ) AS MESSAGES_WITH_RAG_IN_CONVERSATION,

        SUM(
            RAG_RETRIEVAL_COUNT
        ) AS RAG_RETRIEVAL_EVENTS_IN_CONVERSATION

    FROM structured_messages

    GROUP BY
        CONVERSATION_ID
)

SELECT
    sm.CUSTOMER_PHONE,
    sm.CUSTOMER_NAME,

    ec.TOTAL_VISIBLE_MESSAGES_IN_APPENDED_JOURNEY,
    ec.CUSTOMER_MESSAGE_COUNT_IN_APPENDED_JOURNEY,
    ec.AGENT_MESSAGE_COUNT_IN_APPENDED_JOURNEY,

    ec.MESSAGES_WITH_RAG_RETRIEVAL,
    ec.TOTAL_RAG_RETRIEVAL_EVENTS,

    ec.APPENDED_CONVERSATION_COUNT,
    ec.APPENDED_CONVERSATION_IDS,

    ec.TOTAL_CONVERSATIONS_IN_JOURNEY,
    ec.ALLOWED_RESOLVER_CONVERSATIONS,
    ec.OTHER_SKILL_CONVERSATIONS,
    ec.CONVERSATIONS_WITH_UNAUTHORIZED_SKILLS,
    ec.OTHER_SKILL_CONVERSATION_RATIO,

    sm.APPENDED_MESSAGE_INDEX,

    sm.CONVERSATION_ID,
    sm.CONVERSATION_START_DATE,
    sm.CONVERSATION_END_DATE,
    sm.CONVERSATION_STATUS,

    sm.INITIAL_SKILL,
    sm.LAST_SKILL,
    sm.JOINED_SKILLS,

    sm.IS_REQUIRED_RESOLVER_CONVERSATION,
    sm.IS_ALLOWED_RESOLVER_CONVERSATION,

    sm.DISTINCT_SKILL_COUNT,
    sm.ALLOWED_SKILL_COUNT,
    sm.UNAUTHORIZED_SKILL_COUNT,
    sm.UNAUTHORIZED_SKILLS,

    sm.CONVERSATION_AGENT_FULL_NAME,
    sm.CONVERSATION_AGENT_LOGIN_NAME,

    sm.MESSAGE_TIME,
    sm.MESSAGE_SKILL,

    sm.HAS_RAG_RETRIEVAL,
    sm.RAG_RETRIEVAL_COUNT,
    sm.RAG_RETRIEVALS,

    sm.CHUNK_TIME,
    sm.CHUNKS_FETCHED,
    sm.CHUNK_JUSTIFICATION,

    sm.SENDER_ROLE,
    sm.RAW_SENDER_ROLE,
    sm.MESSAGE_AGENT_FULL_NAME,
    sm.MESSAGE_TEXT,

    cc.TOTAL_VISIBLE_MESSAGES_IN_CONVERSATION,
    cc.CUSTOMER_MESSAGE_COUNT_IN_CONVERSATION,
    cc.AGENT_MESSAGE_COUNT_IN_CONVERSATION,
    cc.MESSAGES_WITH_RAG_IN_CONVERSATION,
    cc.RAG_RETRIEVAL_EVENTS_IN_CONVERSATION

FROM structured_messages sm

JOIN eligible_customers ec
    ON sm.CUSTOMER_PHONE = ec.CUSTOMER_PHONE

JOIN conversation_counts cc
    ON sm.CONVERSATION_ID = cc.CONVERSATION_ID

ORDER BY
    sm.CUSTOMER_PHONE,
    sm.APPENDED_MESSAGE_INDEX;
