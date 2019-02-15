CREATE DATABASE IF NOT EXISTS `****` DEFAULT CHARACTER SET utf8mb4 DEFAULT COLLATE utf8mb4_unicode_ci;
USE `****`;


DROP TABLE IF EXISTS `ref_types`;
CREATE TABLE `ref_types` (
  `id` smallint unsigned NOT NULL,					
  `name` varchar(40) NOT NULL,						  
  PRIMARY KEY (`id`),
  UNIQUE KEY `id_UNIQUE` (`id`),
  UNIQUE KEY `name_UNIQUE` (`name`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE utf8mb4_unicode_ci;


DROP TABLE IF EXISTS `journal_groups`;
CREATE TABLE `journal_groups` (
  `id` smallint unsigned NOT NULL,					
  `name` varchar(20) NOT NULL,						  
  PRIMARY KEY (`id`),
  UNIQUE KEY `id_UNIQUE` (`id`),
  UNIQUE KEY `name_UNIQUE` (`name`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE utf8mb4_unicode_ci;


DROP TABLE IF EXISTS `journals`;
CREATE TABLE `journals` (
  `id` smallint unsigned NOT NULL,
  `nlmid` varchar(20) NOT NULL,
  `medline_ta` varchar(200) NOT NULL,
  `group_id` smallint unsigned NOT NULL,	
  PRIMARY KEY (`id`),
  UNIQUE KEY `id_UNIQUE` (`id`),
  UNIQUE KEY `nlmid_UNIQUE` (`nlmid`),
  KEY `group_id_idx` (`group_id`),	
  CONSTRAINT `group_id_fk` FOREIGN KEY (`group_id`) REFERENCES `journal_groups` (`id`) ON DELETE NO ACTION ON UPDATE NO ACTION
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE utf8mb4_unicode_ci;


DROP TABLE IF EXISTS `journal_indexing_periods`;
CREATE TABLE `journal_indexing_periods` (
  `journal_id` smallint unsigned NOT NULL,
  `citation_subset` varchar(6) NOT NULL,
  `is_fully_indexed` boolean NOT NULL, 
  `start_year` smallint unsigned NOT NULL,			
  `end_year` smallint unsigned DEFAULT NULL,			
  KEY `journal_id_idx` (`journal_id`),	
  CONSTRAINT `journal_indexing_periods_journal_id_fk` FOREIGN KEY (`journal_id`) REFERENCES `journals` (`id`) ON DELETE NO ACTION ON UPDATE NO ACTION,
  UNIQUE KEY `journal_indexing_periods_unique_key` (`journal_id`, `citation_subset`, `is_fully_indexed`, `start_year`, `end_year`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE utf8mb4_unicode_ci;


DROP TABLE IF EXISTS `articles`;
CREATE TABLE `articles` (
  `id` int unsigned NOT NULL,
  `pmid` int unsigned NOT NULL,
  `title` varchar(3000) NOT NULL,
  `abstract` varchar(13000) NOT NULL,
  `pub_year` smallint unsigned NOT NULL,			
  `date_completed` DATE NOT NULL,					
  `journal_id` smallint unsigned DEFAULT NULL,
  `is_indexed` boolean NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `id_UNIQUE` (`id`),
  UNIQUE KEY `pmid_UNIQUE` (`pmid`),
  KEY `journal_id_idx` (`journal_id`),	
  CONSTRAINT `articles_journal_id_fk` FOREIGN KEY (`journal_id`) REFERENCES `journals` (`id`) ON DELETE NO ACTION ON UPDATE NO ACTION
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE utf8mb4_unicode_ci;


DROP TABLE IF EXISTS `article_ref_types`;
CREATE TABLE `article_ref_types` (
  `article_id` int unsigned NOT NULL, 
  `ref_type_id` smallint unsigned NOT NULL,  
  KEY `article_id_idx` (`article_id`),
  KEY `ref_type_id_idx` (`ref_type_id`),
  CONSTRAINT `article_id_fk` FOREIGN KEY (`article_id`) REFERENCES `articles` (`id`) ON DELETE NO ACTION ON UPDATE NO ACTION,
  CONSTRAINT `ref_type_id_fk` FOREIGN KEY (`ref_type_id`) REFERENCES `ref_types` (`id`) ON DELETE NO ACTION ON UPDATE NO ACTION
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE utf8mb4_unicode_ci;