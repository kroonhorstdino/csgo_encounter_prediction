const fs = require("fs");
const demofile = require("demofile");
const path = require("path");
const config = require("../config/dataset_config.json");

/** 
 * Data of all weapons, needed for one hot encoding
 * Taken from demofile source code and removed unneeded entries (knifes, etc.)
 */
const itemDefinitionIndexMap = {
    1: {
        itemName: "Desert Eagle",
        className: "weapon_deagle"
    },
    2: {
        itemName: "Dual Berettas",
        className: "weapon_elite"
    },
    3: {
        itemName: "Five-SeveN",
        className: "weapon_fiveseven"
    },
    4: {
        itemName: "Glock-18",
        className: "weapon_glock"
    },
    7: {
        itemName: "AK-47",
        className: "weapon_ak47"
    },
    8: {
        itemName: "AUG",
        className: "weapon_aug"
    },
    9: {
        itemName: "AWP",
        className: "weapon_awp"
    },
    10: {
        itemName: "FAMAS",
        className: "weapon_famas"
    },
    11: {
        itemName: "G3SG1",
        className: "weapon_g3sg1"
    },
    13: {
        itemName: "Galil AR",
        className: "weapon_galilar"
    },
    14: {
        itemName: "M249",
        className: "weapon_m249"
    },
    16: {
        itemName: "M4A4",
        className: "weapon_m4a1"
    },
    17: {
        itemName: "MAC-10",
        className: "weapon_mac10"
    },
    19: {
        itemName: "P90",
        className: "weapon_p90"
    },
    23: {
        itemName: "MP5-SD",
        className: "weapon_mp5sd"
    },
    24: {
        itemName: "UMP-45",
        className: "weapon_ump45"
    },
    25: {
        itemName: "XM1014",
        className: "weapon_xm1014"
    },
    26: {
        itemName: "PP-Bizon",
        className: "weapon_bizon"
    },
    27: {
        itemName: "MAG-7",
        className: "weapon_mag7"
    },
    28: {
        itemName: "Negev",
        className: "weapon_negev"
    },
    29: {
        itemName: "Sawed-Off",
        className: "weapon_sawedoff"
    },
    30: {
        itemName: "Tec-9",
        className: "weapon_tec9"
    },
    31: {
        itemName: "Zeus x27",
        className: "weapon_taser"
    },
    32: {
        itemName: "P2000",
        className: "weapon_hkp2000"
    },
    33: {
        itemName: "MP7",
        className: "weapon_mp7"
    },
    34: {
        itemName: "MP9",
        className: "weapon_mp9"
    },
    35: {
        itemName: "Nova",
        className: "weapon_nova"
    },
    36: {
        itemName: "P250",
        className: "weapon_p250"
    },
    38: {
        itemName: "SCAR-20",
        className: "weapon_scar20"
    },
    39: {
        itemName: "SG 553",
        className: "weapon_sg556"
    },
    40: {
        itemName: "SSG 08",
        className: "weapon_ssg08"
    },
    42: {
        itemName: "Knife",
        className: "weapon_knife"
    },
    43: {
        itemName: "Flashbang",
        className: "weapon_flashbang"
    },
    44: {
        itemName: "High Explosive Grenade",
        className: "weapon_hegrenade"
    },
    45: {
        itemName: "Smoke Grenade",
        className: "weapon_smokegrenade"
    },
    46: {
        itemName: "Molotov",
        className: "weapon_molotov"
    },
    47: {
        itemName: "Decoy Grenade",
        className: "weapon_decoy"
    },
    48: {
        itemName: "Incendiary Grenade",
        className: "weapon_incgrenade"
    },
    49: {
        itemName: "C4 Explosive",
        className: "weapon_c4"
    },
    59: {
        itemName: "Knife",
        className: "weapon_knife_t"
    },
    60: {
        itemName: "M4A1-S",
        className: "weapon_m4a1_silencer"
    },
    61: {
        itemName: "USP-S",
        className: "weapon_usp_silencer"
    },
    63: {
        itemName: "CZ75-Auto",
        className: "weapon_cz75a"
    },
    64: {
        itemName: "R8 Revolver",
        className: "weapon_revolver"
    },
    500: {
        itemName: "Bayonet",
        className: "weapon_bayonet"
    },
    505: {
        itemName: "Flip Knife",
        className: "weapon_knife_flip"
    },
    506: {
        itemName: "Gut Knife",
        className: "weapon_knife_gut"
    },
    507: {
        itemName: "Karambit",
        className: "weapon_knife_karambit"
    },
    508: {
        itemName: "M9 Bayonet",
        className: "weapon_knife_m9_bayonet"
    },
    509: {
        itemName: "Huntsman Knife",
        className: "weapon_knife_tactical"
    },
    512: {
        itemName: "Falchion Knife",
        className: "weapon_knife_falchion"
    },
    514: {
        itemName: "Bowie Knife",
        className: "weapon_knife_survival_bowie"
    },
    515: {
        itemName: "Butterfly Knife",
        className: "weapon_knife_butterfly"
    },
    516: {
        itemName: "Shadow Dag",
        className: "weapon_knife_push"
    },
    519: {
        itemName: "Ursus Knife",
        className: "weapon_knife_ursus"
    },
    520: {
        itemName: "Navaja Knife",
        className: "weapon_knife_gypsy_jackknife"
    },
    522: {
        itemName: "Stiletto Knife",
        className: "weapon_knife_stiletto"
    },
    523: {
        itemName: "Talon Knife",
        className: "weapon_knife_widowmaker"
    }
};


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

        // Object that writes to CSV file
        this.featureWriterObject = this.createFeatures();
        this.featureWriterObject.path = path.join(targetDirectory, path.basename(demoFilePath, ".dem") + '.csv');

        this.createCsvWriter = require('csv-writer').createObjectCsvWriter;

        this.playersInTeams = [];

        this.deathDataBuffer = [];
        this.playerInfoBuffer = [];

        if (this.verbosity > 0) console.log("$$$$$$ Attempting to parse " + this.demoFileName);
        if (this.verbosity > 0) console.log("$$$$$$ Target path set to: " + this.featureWriterObject.path);

        //Delete file before wrtiting a new one
        try {
            fs.unlinkSync(this.featureWriterObject.path);
        } catch (e) {

        } finally {
            try {
                this.subscribeToDemoEvents();
                this.parseDemoFile();
            } catch (e) {
                console.log(">>>> Error message during demo parsing: \n>>>>>> " + e.message);
                console.log(">>>> Demo has some irregularities, aborting and deleting file!");

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

            //Put players into an objects that holds both teams
            for (const team of this.demoFile.teams) {
                this.playersInTeams.push({
                    index: team.index,
                    players: team.members,
                });
            }
        });

        //Get player info at each relevant tick
        this.demoFile.on("tickend", tick => this.on_tickend());

        this.demoFile.gameEvents.on("round_start", s => this.start_ignore_ticks());
        this.demoFile.gameEvents.on("round_freeze_end", f => this.start_parsing_ticks());

        //Write down data at the end of every Round
        this.demoFile.gameEvents.on("round_officially_ended", s => this.on_round_officially_ended());

        //End of the demo file
        this.demoFile.on("end", e => {
            this.on_round_officially_ended();
            if (this.verbosity > 0) console.log(`------------ Parsing of demo ${this.demoFileName} is complete`);


            let millis = Date.now() - this.startTime;
            if (this.verbosity > 1) console.log("------------ Parsing of demo took " + Math.floor(millis / 1000) + " seconds");
            if (this.verbosity > 2) console.log(`------------ ${this.sucessfulParsedCounter} of ${this.demoFile.currentTick} ticks have been parsed and saved`);
        });

        if (this.verbosity > 1) console.log("$$$$$$ Succesfully subscribed to all events!");
    }

    parseDemoFile() {
        this.startTime = Date.now();

        /**
         * Final writer for parsing this demo
         */
        this.writerToPlayerInfo = this.createCsvWriter(this.featureWriterObject);

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
            console.log("|| Tick: " + this.demoFile.currentTick + " | " + this.sucessfulParsedCounter);
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

        if (this.verbosity > 2) console.log(`===========> ${this.demoFileName} - ROUND: ${finishedRound} | TICK: ${this.demoFile.currentTick} PARSED!`);

        /** 
         * Write round intro all stored entries
         */
        this.deathDataBuffer.forEach((kill) => {
            kill.round = finishedRound;
        });
        this.playerInfoBuffer.forEach((playerInfo) => {
            playerInfo.round = finishedRound;
        });

        /** 
         * WRITE DATA TO CSV FILE
         */
        this.writerToPlayerInfo
            .writeRecords(this.playerInfoBuffer);
        //this.writerToDeaths.writeRecords(this.deathDataBuffer);
        if (this.verbosity > 2) console.log("===========> Data written to CSV");

        this.playerInfoBuffer.length = 0;
        this.deathDataBuffer.length = 0;
    }

    //Should a tick message be printed now?
    is_show_print_message() {
        return (this.verbosity > 3) && (this.tickCounter % 1000 == 0);
    }

    /**
     * Is this tick relevant for parsing?
     *
     * @returns
     * @memberof DemoFileParser
     */
    isRelevantTick() {
        return !this.ignoreTicks && (this.tickCounter % this.SampleRateModulo == 0);
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
     * Parses information of all players for current tick and adds it to info array for current round
     *
     * @memberof DemoFileParser
     */
    getAllPlayerInfoForTick() {

        //Terrorist
        let t = this.demoFile.teams[2];
        //Counter-Terrorist
        let ct = this.demoFile.teams[3];

        //If anything is wrong with the teams
        if (t == undefined || ct == undefined) return;


        const allPlayers = [].concat(t.members, ct.members);

        const currentTick = this.demoFile.currentTick;

        if (allPlayers.length != 10 || allPlayers.includes(null)) {
            if (this.is_show_print_message()) console.log("|| Tick: " + currentTick + " | Teams are not complete, discarding this tick!");
            return;
        }

        //Info from current tick for all players
        let tickInfo = {
            tick: currentTick
        };

        //Player counter for later loop
        let playerCounter = 0;

        for (const player of allPlayers) {

            //Go through all players and get features

            const featureNameStart = `f_${playerCounter}`;

            const playerID = player ? player.userId : "-1";
            const playerName = player ? player.name : "playerMissing";


            const position = player.position;

            tickInfo[featureNameStart + "_playerID"] = playerID; //TODO Not used as inuput

            tickInfo[featureNameStart + "_isAlive"] = player.isAlive ? 1 : 0;

            tickInfo[featureNameStart + "_positionX"] = position.x;
            tickInfo[featureNameStart + "_positionY"] = position.y;
            tickInfo[featureNameStart + "_positionZ"] = position.z;

            tickInfo[featureNameStart + "_velocityX"] = player.velocity.x;
            tickInfo[featureNameStart + "_velocityY"] = player.velocity.y;
            tickInfo[featureNameStart + "_velocityZ"] = player.velocity.z;

            //Health, Armor, etc.
            tickInfo[featureNameStart + "_health"] = player.health;
            tickInfo[featureNameStart + "_armor"] = player.armor;

            //Tactical information
            tickInfo[featureNameStart + "_isSpotted"] = player.isSpotted ? 1 : 0;
            tickInfo[featureNameStart + "_isScoped"] = player.isScoped ? 1 : 0;
            tickInfo[featureNameStart + "_isDefusing"] = player.isDefusing ? 1 : 0;

            /** 
             * WEAPONS
             */

            if (player.weapon != null) {
                tickInfo[`${featureNameStart}_currentWeapon`] = this.featureGetCorrectWeaponIndex(player.weapon.itemIndex);
            }

            let allyCounter = 1; //The player itself has index 0.
            let enemyCounter = 0;

            //Distance to other players (Index begins at 1 for allies, 0 for enemies)
            for (const otherPlayer of allPlayers) {

                //Don't measure distance to yourself!
                if (otherPlayer.userId == player.userId) {
                    continue;
                }

                //Calc distance to other player
                let distanceToOther = this.calcVectorDistance(player.position, otherPlayer.position);

                //Other player is in ally team
                if (player.teamNumber == otherPlayer.teamNumber) {
                    tickInfo[`${featureNameStart}_distanceToAlly_${allyCounter}`] = distanceToOther;
                    allyCounter++;
                }
                //Other player on enemy team (Ally_0 is player itself)
                else {
                    tickInfo[`${featureNameStart}_distanceToEnemy_${enemyCounter}`] = distanceToOther;
                    enemyCounter++;
                }
            }

            playerCounter++;
        }

        //Check if anything went wrong
        if (Object.values(tickInfo).includes(null) || Object.values(tickInfo).includes(NaN)) {
            if (this.verbosity > 3) console.log("|| Tick: " + currentTick + " | Something went wrong, discarding this tick!");
            return;
        } else {
            this.sucessfulParsedCounter++;
            this.playerInfoBuffer = this.playerInfoBuffer.concat(tickInfo);
        }
    }

    createFeatures() {

        const playerFeatures = [
            /*{
                    id: 'playerName',
                    title: 'PlayerName'
                },*/
            {
                id: 'isAlive',
                title: 'IsAlive'
            }, {
                id: 'positionX',
                title: 'PositionX'
            }, {
                id: 'positionY',
                title: 'PositionY'
            }, {
                id: 'positionZ',
                title: 'PositionZ'
            }, {
                id: 'velocityX',
                title: 'VelocityX'
            }, {
                id: 'velocityY',
                title: 'VelocityY'
            }, {
                id: 'velocityZ',
                title: 'VelocityZ'
            },
            //Relative distance to all other players (1-9)
            {
                id: 'distanceToAlly_1',
                title: 'DistanceToAlly_1'
            }, {
                id: 'distanceToAlly_2',
                title: 'DistanceToAlly_2'
            }, {
                id: 'distanceToAlly_3',
                title: 'DistanceToAlly_3'
            }, {
                id: 'distanceToAlly_4',
                title: 'DistanceToAlly_4'
            }, {
                id: 'distanceToEnemy_0',
                title: 'DistanceToEnemy_0'
            }, {
                id: 'distanceToEnemy_1',
                title: 'DistanceToEnemy_1'
            }, {
                id: 'distanceToEnemy_2',
                title: 'DistanceToEnemy_2'
            }, {
                id: 'distanceToEnemy_3',
                title: 'DistanceToEnemy_3'
            }, {
                id: 'distanceToEnemy_4',
                title: 'DistanceToEnemy_4'
            },
            //Health, etc.
            {
                id: 'health',
                title: 'Health'
            }, {
                id: 'isScoped',
                title: 'IsScoped'
            }, {
                id: 'isDefusing',
                title: 'IsDefusing'
            },
            {
                id: 'currentWeapon',
                title: 'CurrentWeapon'
            }
            /*{
                   id: '',
                   title:
               }*/
        ];

        /*
        TODO: Hot one encoding somewhere. Do it afterwards in  preprocessing
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

        let allFeatures = [{
                id: 'round',
                title: 'Round'
            },
            {
                id: 'tick',
                title: 'Tick'
            }
        ];

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
     * @param {*} weaponIndex
     * @returns {boolean}
     * @memberof DemoFileParser
     */
    isDifferentKnifeWeaponIndex(weaponIndex) {
        return (weaponIndex >= 500 || weaponIndex == 59);
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
                    return Math.round(this.demoFile.tickRate / (this.demoFile.tickRate / 8)); //64 / (64 / 8)

            }
        } catch (e) {
            if (this.verbosity > 1) console.log("$$$$$$ No tickrate detected! Assuming demo tickrate of 32");
            return 4; //New tick has been seen
        }
    }

    /**
     * Calculates distance between two values
     *
     * @param {*} vecA
     * @param {*} vecB
     * @param {boolean} useTaxicab If true, the cheaper taxicab algorithm is used.
     * @returns
     * @memberof DemoFileParser
     */
    calcVectorDistance(vecA, vecB, useTaxicab = true) {
        //TODO Use pythagoras!
        if (useTaxicab) {
            let dX = Math.abs(vecA.x - vecB.x);
            let dY = Math.abs(vecA.y - vecB.y);
            let dZ = Math.abs(vecA.z - vecB.z);

            return dX + dY + dZ;
        } else {}
    }
}

let demoFilePath, parsedFilePath;
let verbosity = 4;

//Its weird, but it works
if (process.argv[2] == "true") {
    demoFilePath = '../demo_files/sprout-vs-ex-epsilon-m3-overpass.dem';
    parsedFilePath = 'parsed_files/';
} else {
    //TODO different ways to set paths
    demoFilePath = process.argv[2];
    parsedFilePath = process.argv[3];
    verbosity = Number(process.argv[4]);
}

console.log("Start parsing...");
new DemoFileParser(demoFilePath, parsedFilePath, verbosity);

/*
try {
    fs.unlinkSync('parsed_files\\positions.csv')
    console.log("Deleted old file!")
} catch (e) {
    console.log("Do nothing!")
}
*/

//demo.parseDemoFile(path.resolve('vitality-vs-liquid-m2-dust2.dem'))