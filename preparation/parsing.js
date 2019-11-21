const fs = require("fs");
const demofile = require("demofile");
//const p5 = require("p5");

const path = require("path");

const config = require("../config/dataset_config.json");
const featuresInfoList = require("../config/features_info.json");
//const itemDefinitionIndexMapConfig = require("../config/item_definition_index_map.json")

/**
 * Data of all weapons, needed for one hot encoding
 * Taken from demofile source code and removed unneeded entries (knifes, etc.)
 */
//const itemDefinitionIndexMap = itemDefinitionIndexMapConfig["itemDefinitionIndexMap"];

class DemoFileParser {
    constructor(demoFilePath, targetDirectory, verbosity = 1) {
        this.demoFile = new demofile.DemoFile();

        this.verbosity = verbosity;
        this.demoFilePath = demoFilePath;
        this.demoFileName = path.basename(demoFilePath);

        //How many ticks have actually been parsed
        this.sucessfulParsedCounter = 0;

        //How many ticks have been seen as of now?
        this.tickCounter = -1;

        this.deathWriterObject = {};
        this.deathWriterObject.header = [...featuresInfoList["demo_features"]]
        this.deathWriterObject.header.push(
            {
                "id": "attackerIndex",
                "title": "AttackerIndex"
            },
            {
                "id": "attackerName",
                "title": "AttackerName"
            },
            {
                "id": "victimIndex",
                "title": "VictimIndex"
            },
            {
                "id": "victimName",
                "title": "VictimName"
            }
        )
        this.deathWriterObject.path = path.join(
            targetDirectory,
            path.basename(demoFilePath, ".dem") + "_deaths.csv"
        );

        // Object that writes to CSV file
        this.featureWriterObject = this.createFeatures();
        this.featureWriterObject.path = path.join(
            targetDirectory,
            path.basename(demoFilePath, ".dem") + ".csv"
        );

        this.createCsvWriter = require("csv-writer").createObjectCsvWriter;

        this.deathDataBuffer = [];
        this.playerInfoBuffer = [];

        if (this.verbosity > 0)
            console.log("$$$$$$ Attempting to parse " + this.demoFileName);
        if (this.verbosity > 0)
            console.log(
                "$$$$$$ Target path set to: " + this.featureWriterObject.path
            );

        //Delete file before wrtiting a new one
        try {
            fs.unlinkSync(this.featureWriterObject.path);
            fs.unlinkSync(this.deathWriterObject.path);
        } catch (e) {
        } finally {
            try {
                this.subscribeToDemoEvents();
                this.parseDemoFile();
            } catch (e) {
                console.log(
                    ">>>> Error message during demo parsing: \n>>>>>> " + e.message
                );
                console.log(
                    ">>>> Demo has some irregularities, aborting and deleting file!"
                );

                process.exitCode = 1;
            }
        }
    }

    /**
     *
     *
     * @memberof DemoFileParser
     */
    subscribeToDemoEvents() {
        //Setting up things
        this.demoFile.on("start", s => {
            this.SampleRateModulo = this.getTickSampleRateModulo();

            this.ignoreTicks = true; //Ingore ticks until first round starts
        });

        this.demoFile.gameEvents.on("player_death", e => {

            const deathEventInfo = this.getDeathInfo(e)
            if (Object.values(deathEventInfo).includes("unnamed") || deathEventInfo["victimIndex"] == -1 || Object.values(deathEventInfo).includes(null) ||
                Object.values(deathEventInfo).includes(NaN)) {
                console.log("Discard death event tick!")
            }
            else {
                this.deathDataBuffer.push(deathEventInfo);
            }
        });

        //Get player info at each relevant tick
        this.demoFile.on("tickend", tick => this.on_tickend());

        this.demoFile.gameEvents.on("round_start", s => this.start_ignore_ticks());
        this.demoFile.gameEvents.on("round_freeze_end", f =>
            this.start_parsing_ticks()
        );

        //Write down data at the end of every Round
        this.demoFile.gameEvents.on("round_officially_ended", s =>
            this.on_round_officially_ended()
        );

        //End of the demo file
        this.demoFile.on("end", e => {
            this.on_round_officially_ended();
            if (this.verbosity > 0)
                console.log(
                    `------------ Parsing of demo ${this.demoFileName} is complete`
                );

            let millis = Date.now() - this.startTime;
            if (this.verbosity > 1)
                console.log(
                    "------------ Parsing of demo took " +
                    Math.floor(millis / 1000) +
                    " seconds"
                );
            if (this.verbosity > 2)
                console.log(
                    `------------ ${this.sucessfulParsedCounter} of ${this.demoFile.currentTick} ticks have been parsed and saved`
                );
        });

        if (this.verbosity > 1)
            console.log("$$$$$$ Succesfully subscribed to all events!");
    }

    getDeathInfo(e) {
        this.set_teams_in_order(true)

        const players = this.get_all_players_in_order(true)//.map((player => player.userId));

        const victim = this.demoFile.entities.getByUserId(e.userid);
        const victimIndex = victim ? players.findIndex((player => player.userId == victim.userId)) : -1;
        const victimName = victim ? victim.name : "unnamed";

        // Attacker may have disconnected so be aware.
        // e.g. attacker could have thrown a grenade, disconnected, then that grenade
        // killed another player.
        const attacker = this.demoFile.entities.getByUserId(e.attacker);
        const attackerIndex = attacker ? players.findIndex((player => player.userId == attacker.userId)) : -1;
        const attackerName = attacker ? attacker.name : "unnamed";

        //const headshotText = e.headshot ? " HS" : "";

        const deathEventInfo = {
            "time": this.demoFile.currentTime,
            "tick": this.demoFile.currentTick,
            "attackerIndex": attackerIndex,
            "attackerName": attackerName,
            "victimIndex": victimIndex,
            "victimName": victimName,
        }

        return deathEventInfo
    }

    parseDemoFile() {
        this.startTime = Date.now();

        /**
         * Final writer for parsing this demo
         */
        this.writerToPlayerInfo = this.createCsvWriter(this.featureWriterObject);
        this.writerToDeathInfo = this.createCsvWriter(this.deathWriterObject);

        let buffer = null;

        try {
            buffer = fs.readFileSync(this.demoFilePath);
        } catch (e) {
            console.error("(`------------ Demo file could not be opened! " + e);
            process.exitCode = 1;
            return;
        }

        this.demoFile.parse(buffer);
    }

    on_tickend() {
        this.tickCounter++; //New tick has been seen
        /**
         * If this tick is not relevant, do not parse this tick
         */
        if (!this.isRelevantTick()) return;

        if (this.is_show_print_message()) {
            console.log(
                "|| Tick: " +
                this.demoFile.currentTick +
                " | " +
                this.sucessfulParsedCounter
            );
        }

        /**
         * Get player positions each tick
         */
        this.getAllPlayerInfoForTick();
    }

    /**
     * The very end of the round, including time after timer ends
     *
     * @param {*} e
     * @memberof DemoFileParser
     */
    on_round_officially_ended() {
        const teams = this.demoFile.teams;

        //const terrorists = teams[2];
        //const cts = teams[3];

        const finishedRound = this.demoFile.gameRules.roundsPlayed;

        if (this.verbosity > 2)
            console.log(
                `===========> ${this.demoFileName} - ROUND: ${finishedRound} | TICK: ${this.demoFile.currentTick} PARSED!`
            );

        /**
         * Write round intro all stored entries
         */
        this.deathDataBuffer.forEach(kill => {
            kill.round = finishedRound;
        });
        this.playerInfoBuffer.forEach(playerInfo => {
            playerInfo.round = finishedRound;
        });

        /**
         * WRITE DATA TO CSV FILE
         */
        this.writerToPlayerInfo.writeRecords(this.playerInfoBuffer);
        this.writerToDeathInfo.writeRecords(this.deathDataBuffer);
        if (this.verbosity > 2) console.log("===========> Data written to CSV");

        this.playerInfoBuffer.length = 0;
        this.deathDataBuffer.length = 0;
    }

    /**
     * Returns an array of teams sorted by team handle
     * Teams are always in same order regardless of which side they're on (CT or T)!
     *
     * @param {*} reset
     * @returns
     * @memberof DemoFileParser
     */
    set_teams_in_order(reset) {

        //Terrorist
        let t = this.demoFile.teams[2];
        //Counter-Terrorist
        let ct = this.demoFile.teams[3];

        let hasReset = false;

        try {
            if (this.teams == null || reset || this.teams == undefined) {
                // Sort teams by handle to ensure a constant order of teams and players
                let allTeams = [].concat(t, ct).sort((a, b) => a.clanName.localeCompare(b.clanName));
                this.teams = allTeams;
                if (allTeams != this.teams) hasReset = true;
                //console.log(allTeams[0].clanName)
            }
        } catch (e) {
            // DO nothing
        }

        return hasReset;
    }

    /**
     * Returns an array of all players in order of teams first and then players ordered by index inside the teams
     *
     * @returns
     * @memberof DemoFileParser
     */
    get_all_players_in_order(reset) {
        try {
            if (this.players_in_order == null || reset) {

                this.players_in_order = [].concat(this.teams[0].members.sort((a, b) => a.name.localeCompare(b.name)), this.teams[1].members.sort((a, b) => a.name.localeCompare(b.name)));
            }
        } catch (e) {
            console.log("No teams stored!")
        }

        return this.players_in_order
    }



    //Should a tick message be printed now?
    is_show_print_message() {
        return this.verbosity > 3 && this.tickCounter % 1000 == 0;
    }

    /**
     * Is this tick relevant for parsing?
     *
     * @returns
     * @memberof DemoFileParser
     */
    isRelevantTick() {
        return !this.ignoreTicks && this.tickCounter % this.SampleRateModulo == 0 && !this.demoFile.gameRules.isWarmup && ['first', 'second'].includes(this.demoFile.gameRules.phase);
    }

    /**
     * Very start of the round, including buy time
     *
     * @memberof DemoFileParser
     */
    start_ignore_ticks() {
        this.ignoreTicks = true;
    }

    /**
     * Once the timer of the round starts and players can move
     *
     */
    start_parsing_ticks() {
        //Start parsing ticks once player can move again
        this.ignoreTicks = false;
    }

    /**
     * ########
     * COLLECT INFORMATION
     * ########
     */

    /**
     * Parses information of all players for current tick and adds it to info array for current round
     *
     * @memberof DemoFileParser
     */
    getAllPlayerInfoForTick() {
        const hasReset = this.set_teams_in_order(true);
        const allPlayers = this.get_all_players_in_order(true);

        const currentTick = this.demoFile.currentTick;

        if (allPlayers.length != 10 || allPlayers.includes(null) || allPlayers == undefined) {
            if (this.is_show_print_message())
                console.log(
                    "|| Tick: " +
                    currentTick +
                    " | Teams are not complete, discarding this tick!"
                );
            return;
        }

        //Stores info about all players from current tick
        let tickInfo = {
            time: this.demoFile.currentTime,
            tick: currentTick
        };

        //Player counter for later loop
        let playerCounter = 0;

        /**
         * @type {demofile.Player} player
         */
        for (const player of allPlayers) {
            //Go through all players and get features

            const featureNameStart = `f_${playerCounter}`;

            const playerID = player ? player.userId : "-1";
            //const playerName = player ? player.name : "playerMissing";

            const position = player.position;
            //const p5originPos = p5.createVector(playerPos.x, playerPos.y, playerPos.z);


            tickInfo[featureNameStart + "_playerID"] = playerID; //TODO Not used as inuput

            tickInfo[featureNameStart + "_isAlive"] = player.isAlive ? 1 : 0;

            tickInfo[featureNameStart + "_positionX"] = position.x;
            tickInfo[featureNameStart + "_positionY"] = position.y;
            tickInfo[featureNameStart + "_positionZ"] = position.z;

            tickInfo[featureNameStart + "_velocityX"] = player.velocity.x;
            tickInfo[featureNameStart + "_velocityY"] = player.velocity.y;
            tickInfo[featureNameStart + "_velocityZ"] = player.velocity.z;

            tickInfo[featureNameStart + "_eyeAnglePitch"] = player.eyeAngles.pitch;
            tickInfo[featureNameStart + "_eyeAngleYaw"] = player.eyeAngles.yaw;

            //Health, Armor, etc.
            tickInfo[featureNameStart + "_health"] = player.health;
            tickInfo[featureNameStart + "_armor"] = player.armor;

            //Tactical information
            tickInfo[featureNameStart + "_hasHelmet"] = player.hasHelmet ? 1 : 0;
            tickInfo[featureNameStart + "_isSpotted"] = player.isSpotted ? 1 : 0;
            tickInfo[featureNameStart + "_isScoped"] = player.isScoped ? 1 : 0;
            tickInfo[featureNameStart + "_isDefusing"] = player.isDefusing ? 1 : 0;

            // TODO: String bad! Find a solution for preprocessing. Must be one hot encoded?
            tickInfo[featureNameStart + "_placeName"] = player.placeName;

            /**
             * ########
             * WEAPONS
             * ########
             */

            tickInfo[featureNameStart + "_equipmentValue"] = player.currentEquipmentValue;

            if (player.weapon != null) {
                tickInfo[
                    `${featureNameStart}_currentWeapon`
                ] = this.featureGetCorrectWeaponIndex(player.weapon.itemIndex);
            } else {
            }

            /**
             * ###############
             * VALUES RELATIVE TO OTHER PLAYERS
             * ###############
             */

            let allyCounter = 1; //The player itself has index 0, so we begin at 1.
            let enemyCounter = 0;

            //Distance to other players (Index begins at 1 for allies, 0 for enemies)
            for (const otherPlayer of allPlayers) {
                //Don't measure distance to yourself!
                if (otherPlayer.userId == player.userId) {
                    continue;
                }

                const otherPosition = otherPlayer.position;
                //targetPos = p5.createVector(otherPos.x, otherPos.y, otherPos.z);

                //Calc distance to other player
                let distanceToOther = this.calcVectorDistance(
                    player.position,
                    otherPosition
                );


                //Other player is in ally team
                if (player.teamNumber == otherPlayer.teamNumber) {
                    tickInfo[
                        `${featureNameStart}_distanceToAlly_${allyCounter}`
                    ] = distanceToOther;
                    allyCounter++;
                }
                //Other player on enemy team (Ally_0 is player itself)
                else {
                    tickInfo[
                        `${featureNameStart}_distanceToEnemy_${enemyCounter}`
                    ] = distanceToOther;

                    /** 
                    let angleToOther = null
                    tickInfo[
                        `${featureNameStart}_angleAbsoluteToEnemy_${enemyCounter}`
                    ] = this.angleToOther(p5originPos, player.eyeAngles, otherPosition);
                    */

                    enemyCounter++;
                }
            }

            playerCounter++;
        }

        //Check if anything went wrong
        if (
            Object.values(tickInfo).includes("unnamed") || Object.values(tickInfo).includes(-1) || Object.values(tickInfo).includes(null) ||
            Object.values(tickInfo).includes(NaN)) {
            if (this.verbosity > 0)
                console.warn(
                    "|| Tick: " +
                    currentTick +
                    " | Something went wrong, discarding this tick!"
                );
            return;
        } else {
            this.sucessfulParsedCounter++;
            this.playerInfoBuffer = this.playerInfoBuffer.concat(tickInfo);
        }
    }

    /**
     * Calculates distance between two X,Y positions
     *
     * @param {Vector} positionA
     * @param {Vector} positionB
     * @param {boolean} useTaxicab If true, the cheaper taxicab algorithm is used.
     * @returns {Number} A distance
     * @memberof DemoFileParser
    */
    calcVectorDistance(positionA, positionB) {
        //Copied form https://gist.github.com/timohausmann/5003280
        let xs = positionA.x - positionB.x,
            ys = positionA.y - positionB.y;

        // Square values
        xs *= xs;
        ys *= ys;

        // Pythagoras
        return Math.sqrt(xs + ys);
    }

    /**
     * Get angle of croshair to other player
     * 
     * @param {Vector} playerPos 
     * @param {Object{pitch, yaw}} playerAngles 
     * @param {Vector} otherPos 
     */
    angleToOther(playerPos, lookingVec, otherPos) {

        //directionVec = p5.Vector.sub(targetPos, originPos);

        //directionVec.angleBetween()

        return 0;
    }

    createFeatures() {
        let playerFeatures = [];
        for (const featureSetName of featuresInfoList["player_features_sets"]["parse_correct_all"]) {
            playerFeatures = playerFeatures.concat(featuresInfoList["player_features"][featureSetName]);
        }

        /*
            DONE: Hot one encoding somewhere. Do it afterwards in  preprocessing
            for (const weaponIndex in itemDefinitionIndexMap) {
                //Don't add other knifes to the feature list, all knifes will be mapped to "normal knife" (index 42)
                if (this.isDifferentKnifeWeaponIndex(weaponIndex)) continue;
     
                const weaponClassName = `${itemDefinitionIndexMap[weaponIndex].className}_${weaponIndex}`;
     
                playerFeatures.push({
                    id: weaponClassName,
                    //Uppercase for first letter (I don't know why exactly)
                    title: (weaponClassName.substring(0, 1)[0].toUpperCase()).concat(weaponClassName.substring(1))
                });
            }*/

        let allFeatures = [...featuresInfoList["demo_features"]]

        //Generate features per player and push into allFeatures array
        for (const feature of playerFeatures) {
            for (let i = 0; i < 10; i++) {
                let newId = `f_${i}_${feature.id}`;
                let newTitle = `f_${i}_${feature.title}`;

                allFeatures.push({
                    id: newId,
                    title: newTitle
                });
            }
        }

        //Writer object represents features as columns now
        let writerObject = {};
        writerObject.header = allFeatures;

        return writerObject;
    }

    /**
     * Gets weapon index and converts it into correct index if necessary
     * For example: Knifes have different IDs but should be mapped to one index
     *
     * @param {Number} weaponIndex Index of weapon
     * @memberof DemoFileParser
     */
    featureGetCorrectWeaponIndex(weaponIndex) {
        if (this.isDifferentKnifeWeaponIndex(weaponIndex)) {
            //It is a different knife!
            return 42;
            //Maps index of knife to that of the "normal knife"
        }

        return weaponIndex;
    }

    /**
     * Checks if knife is a knife other than the "normal knife" (index 42)
     *
     * Above index 500 only knifes
     * index 59 is t_knife
     *
     * @param {*} weaponIndex
     * @returns {boolean}
     * @memberof DemoFileParser
     */
    isDifferentKnifeWeaponIndex(weaponIndex) {
        return weaponIndex >= 500 || weaponIndex == 59;
    }

    /**
     * Returns which ticks are supposed to be used for parsing.
     *
     * @returns
     * @memberof DemoFileParser
     */
    getTickSampleRateModulo() {
        //Incase tickrate is not accesible in the demo
        try {
            if (!(this.demoFile.tickRate >= 0)) throw NaN;
            switch (this.demoFile.tickRate) {
                case 128:
                    return 16;

                case 64:
                    return 8; //Get every 8th tick

                case 32:
                    return 4; //Get every 4th tick

                case 16:
                    return 2;

                default:
                    //In case of irregular tick rates
                    return Math.round(
                        this.demoFile.tickRate / (this.demoFile.tickRate / 8)
                    ); //64 / (64 / 8)
            }
        } catch (e) {
            if (this.verbosity > 1)
                console.log(
                    "$$$$$$ No tickrate detected! Assuming demo tickrate of 64"
                );
            return 8; //New tick has been seen
        }
    }
}

let demoFilePath, parsedFilePath;
let verbosity = 4;

let debugDemoFilePath = "/home/hueter/csgo_dataset/demo_files_57004/pact-vs-movistar-riders-vertigo.dem"

//Its weird, but it works
if (process.argv[2] == "true") {
    demoFilePath = debugDemoFilePath;
    parsedFilePath = "../../csgo_dataset/";
} else {
    //TODO different ways to set paths
    demoFilePath = process.argv[2];
    parsedFilePath = process.argv[3];
    verbosity = Number(process.argv[4]);
}

console.log("Start parsing...");
new DemoFileParser(demoFilePath, parsedFilePath, verbosity);
