CREATE SCHEMA IF NOT EXISTS `agentsmith`;

CREATE TABLE IF NOT EXISTS `agentsmith`.`account` (
    `id` bigint unsigned NOT NULL,
    `name` VARCHAR(255) NOT NULL,
    `type` VARCHAR(2) NOT NULL,
    `status` TINYINT NOT NULL DEFAULT 0,
    `created_on` DATETIME NOT NULL,
    `created_by` VARCHAR(45) NOT NULL,
    `modified_by` VARCHAR(45) NULL,
    `modified_on` DATETIME NULL,
    `version` INT NOT NULL DEFAULT 0,
    `tenant_id` VARCHAR(50) NOT NULL,
    PRIMARY KEY (`id`))
    ENGINE = InnoDB;

CREATE TABLE IF NOT EXISTS `agentsmith`.`profile` (
    `id` bigint unsigned NOT NULL,
    `eff_from` DATETIME NOT NULL,
    `eff_to` DATETIME NOT NULL,
    `name_prefix` VARCHAR(200) NULL,
    `name_suffix` VARCHAR(200) NULL,
    `givenname` VARCHAR(200) NOT NULL,
    `middlename` VARCHAR(200) NULL,
    `familyname` VARCHAR(200) NOT NULL,
    `known_as` VARCHAR(200) NULL,
    `date_of_birth` DATE NULL,
    `sex_at_birth` VARCHAR(1) NULL,
    `deceased_date` DATE NULL,
    `data` JSON NOT NULL,
    `created_on` DATETIME NOT NULL,
    `created_by` VARCHAR(45) NOT NULL,
    `modified_on` DATETIME NULL,
    `modified_by` VARCHAR(45) NULL,
    `version` INT NOT NULL DEFAULT 0,
    `tenant_id` VARCHAR(50) NOT NULL,
    PRIMARY KEY (`id`, `eff_from`, `eff_to`)
    )
    ENGINE = InnoDB;


CREATE TABLE IF NOT EXISTS `agentsmith`.`profile_email` (
    `id` bigint unsigned NOT NULL,
    `profile_id` bigint unsigned NOT NULL,
    `eff_from` DATETIME NOT NULL,
    `eff_to` DATETIME NOT NULL,
    `email` VARCHAR(500) NULL,
    `order` smallint unsigned NULL,
    `created_on` DATETIME NOT NULL,
    `created_by` VARCHAR(45) NOT NULL,
    `modified_on` DATETIME NULL,
    `modified_by` VARCHAR(45) NULL,
    `version` INT NOT NULL DEFAULT 0,
    `tenant_id` VARCHAR(50) NOT NULL,
    PRIMARY KEY (`id`)
    )
    ENGINE = InnoDB;



CREATE TABLE IF NOT EXISTS `agentsmith`.`profile_external_reference` (
    `id` bigint unsigned NOT NULL,
    `profile_id` bigint unsigned NOT NULL,
    `eff_from` DATETIME NOT NULL,
    `eff_to` DATETIME NOT NULL,
    `system` VARCHAR(200) NULL,
    `reference` VARCHAR(200) NULL,
    `created_on` DATETIME NOT NULL,
    `created_by` VARCHAR(45) NOT NULL,
    `modified_on` DATETIME NULL,
    `modified_by` VARCHAR(45) NULL,
    `version` INT NOT NULL DEFAULT 0,
    `tenant_id` VARCHAR(50) NOT NULL,
    PRIMARY KEY (`id`)
    )
    ENGINE = InnoDB;

CREATE TABLE IF NOT EXISTS `agentsmith`.`account_users` (
    `id` bigint unsigned NOT NULL,
    `profile_id` bigint unsigned NOT NULL,
    `account_id` bigint unsigned NOT NULL,
    `status` TINYINT NOT NULL DEFAULT 0,
    `eff_from` DATETIME NOT NULL,
    `eff_to` DATETIME NOT NULL,
    `created_on` DATETIME NOT NULL,
    `created_by` VARCHAR(45) NOT NULL,
    `modified_by` VARCHAR(45) NULL,
    `modified_on` DATETIME NULL,
    `version` INT NOT NULL DEFAULT 0,
    `tenant_id` VARCHAR(50) NOT NULL,
    PRIMARY KEY (`id`))
    ENGINE = InnoDB;

CREATE TABLE IF NOT EXISTS `agentsmith`.`account_users_roles` (
    `id` bigint unsigned NOT NULL,
    `account_user_id` bigint unsigned NOT NULL,
    `role` VARCHAR(45) NOT NULL,
    `status` TINYINT NOT NULL DEFAULT 0,
    `eff_from` DATETIME NOT NULL,
    `eff_to` DATETIME NOT NULL,
    `created_on` DATETIME NOT NULL,
    `created_by` VARCHAR(45) NOT NULL,
    `modified_by` VARCHAR(45) NULL,
    `modified_on` DATETIME NULL,
    `version` INT NOT NULL DEFAULT 0,
    `tenant_id` VARCHAR(50) NOT NULL,
    PRIMARY KEY (`id`))
    ENGINE = InnoDB;


CREATE TABLE IF NOT EXISTS `agentsmith`.`party` (
    `id` bigint unsigned NOT NULL,
    `type` VARCHAR(2) NOT NULL,
    `status` VARCHAR(10) NOT NULL DEFAULT 'ACTIVE',
    `created_on` DATETIME NOT NULL,
    `created_by` VARCHAR(45) NOT NULL,
    `modified_by` VARCHAR(45) NULL,
    `modified_on` DATETIME NULL,
    `version` INT NOT NULL DEFAULT 0,
    `tenant_id` VARCHAR(50) NOT NULL,
    PRIMARY KEY (`id`))
    ENGINE = InnoDB;


CREATE TABLE IF NOT EXISTS `agentsmith`.`person` (
    `id` bigint unsigned NOT NULL,
    `eff_from` DATETIME NOT NULL,
    `eff_to` DATETIME NOT NULL,
    `name_prefix` VARCHAR(200) NULL,
    `name_suffix` VARCHAR(200) NULL,
    `givenname` VARCHAR(200) NOT NULL,
    `middlename` VARCHAR(200) NULL,
    `familyname` VARCHAR(200) NOT NULL,
    `known_as` VARCHAR(200) NULL,
    `date_of_birth` DATE NULL,
    `sex_at_birth` VARCHAR(1) NOT NULL DEFAULT 'U',
    `deceased_date` DATE NULL,
    `data` JSON NOT NULL,
    `created_on` DATETIME NOT NULL,
    `created_by` VARCHAR(45) NOT NULL,
    `modified_on` DATETIME NULL,
    `modified_by` VARCHAR(45) NULL,
    `version` INT NOT NULL DEFAULT 0,
    `tenant_id` VARCHAR(50) NOT NULL,
    PRIMARY KEY (`id`, `eff_from`, `eff_to`),
    CONSTRAINT `fk_person_party`
    FOREIGN KEY (`id`)
    REFERENCES `agentsmith`.`party` (`id`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION)
    ENGINE = InnoDB;



CREATE TABLE IF NOT EXISTS `agentsmith`.`organisation`
(
    `id`          bigint unsigned NOT NULL,
    `parent_id`   VARCHAR(40)   NULL,
    `eff_from`    DATETIME    NOT NULL,
    `eff_to`      DATETIME    NOT NULL,
    `name`        VARCHAR(45)   NOT NULL,
    `type`        INT           NOT NULL,
    `data` JSON NOT NULL,
    `created_on`  datetime      not null,
    `created_by`  varchar(45)   not null,
    `modified_on` datetime      null,
    `modified_by` varchar(45)   null,
    `version`     int default 0 not null,
    `tenant_id`   varchar(50)   not null,
    PRIMARY KEY (`id`, `eff_from`, `eff_to`),
    CONSTRAINT `fk_organisation_party1`
    FOREIGN KEY (`id`)
    REFERENCES `agentsmith`.`party` (`id`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION
    )
    ENGINE = InnoDB;



# INSERT INTO agentsmith.party (id, type, status, created_on, created_by, modified_by, modified_on, version, tenant_id) VALUES (0, 'P', '0', '2024-04-29 09:06:44', 'kevinbayes', 'kevinbayes', '2024-04-29 09:06:55', 0, 'demo');
# INSERT INTO agentsmith.person (id, eff_from, eff_to, name_prefix, name_suffix, givenname, middlename, familyname, known_as, date_of_birth, sex_at_birth, deceased_date, data, created_on, created_by, modified_on, modified_by, version, tenant_id) VALUES (0, '2024-04-28 09:07:10', '9999-12-31 23:59:59', 'Mr', '', 'Homer', null, 'Simpson', 'Homer', '2000-04-29', 'M', null, '{}', '2024-04-29 09:08:17', 'kevinbayes', '2024-04-29 09:08:17', 'kevinbayes', 0, 'demo');

CREATE TABLE IF NOT EXISTS agentsmith.counter (
    id int unsigned NOT NULL,
    description varchar(50) NOT NULL,
    count bigint unsigned NOT NULL,
    PRIMARY KEY  (id)
) ENGINE=InnoDB;


CREATE PROCEDURE IF NOT EXISTS get_next_id (IN counter bigint)
BEGIN
    UPDATE agentsmith.counter
    SET count = (@agentsmith.counter:=count+1)
    WHERE id = counter;
    SELECT @agentsmith.counter as id;
END;

# INSERT INTO agentsmith.counter (id, `name`, count) VALUES (0, 'profile', 1);

-- # SET @counter = 0;
-- # CALL agentsmith.get_next_id(@counter);

SELECT CURRENT_TIMESTAMP;