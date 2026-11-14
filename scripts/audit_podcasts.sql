BEGIN READ ONLY;

SELECT p."Id", p."Title", p."Url" AS slug
FROM "Podcasts" AS p
JOIN "Statuses" AS s ON s."Id" = p."StatusId"
WHERE p."Published" <= now()
  AND s."Title" = 'Опубликовано'
  AND NOT p."IsClosed" AND NOT p."IsDeleted" AND NOT p."NoIndex"
  AND p."Cost" IS NULL
  AND NOT EXISTS (
      SELECT 1 FROM "PodcastSubscriptions" AS sub WHERE sub."PodcastId" = p."Id"
  )
ORDER BY p."Published";

SELECT count(*) AS file_links, count(f."Id") AS resolved_files
FROM "PodcastFiles" AS pf
LEFT JOIN "Files" AS f ON f."Id" = pf."FileId";

ROLLBACK;
