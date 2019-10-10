const fs = require("fs");
const demofile = require("demofile");
const path = require("path");
const config = require("../config/prep_config.json")

/*
//For writing to file that contains deaths
const writerToDeaths = createCsvWriter({
    path: 'death.csv',
    header: [{
            id: 'round',
            title: 'Round'
        },
        {
            id: 'tick',
            title: 'Tick'
        },
        {
            id: 'victim',
            title: 'Victim'
        },
        {
            id: 'victimPosition',
            title: 'VictimPosition'
        },
        {
            id: 'attacker',
            title: 'Attacker'
        },
        {
            id: 'attackerPosition',
            title: 'AttackerPosition'
        },
    ]
});*/

class DemoFileParser {
    constructor(demoFilePath, targetDirectory) {
        this.demoFile = new demofile.DemoFile();

        this.demoFilePath = demoFilePath; 
        this.demoFileName = path.basename(demoFilePath);

        this.featureWriterObject = this.createFeatures();
        this.featureWriterObject.path = targetDirectory + path.basename(demoFilePath,".dem") + '.csv'

        this.createCsvWriter = require('csv-writer').createObjectCsvWriter;

        console.log("Attempting to parse " + this.demoFileName);

        this.playersInTeams = [];

        this.deathDataBuffer = [];
        this.playerInfoBuffer = [];

        try {
            this.subscribeToDemoEvents();
            this.parseDemoFile();
        } catch(e) {
            console.log(">>>> Error message during demo parsing: \n>>>>>> " + e.message)
            console.log(">>>> Demo has some irregularities, aborting and deleting file!");

            try { fs.unlinkSync(this.featureWriterObject.path); } catch(e) {
            }
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
            /*{
                   id: '',
                   title:
               }*/
        ]
        
        let allFeatures = [{
                id: 'round',
                title: 'Round'
            },
            {
                id: 'tick',
                title: 'Tick'
            }
        ]        
        
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

        let writerObject = {}

        writerObject["header"] = allFeatures;

        return writerObject;
    }

    async parseDemoFile() {
        this.startTime = Date.now();

        /**
        * Final writer for parsing this demo
        */
        this.writerToPlayerInfo = this.createCsvWriter(this.featureWriterObject);

        let buffer = null;

        try {
            buffer = fs.readFileSync(this.demoFilePath);
        } catch (e) {
            console.error("Demo file could not be opened! " + e);
            return;
        }

        this.demoFile.parse(buffer);
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
            switch (this.demoFile.tickRate) {
                case 128:
                    return 16;
                    break;
                case 64:
                    return 8; //Get every 8th tick
                    break;
                case 32:
                    return 4; //Get every 4th tick
                    break;
                case 16:
                    return 2;
                    break;
                default:
                    //In case of irregular tick rates
                    return Math.round(this.demoFile.tickRate / (this.demoFile.tickRate / 8)) //64 / (64 / 8)
                    break;
            }
        } catch (e) {
            console.log("No tickrate!");
        }
        finally {
            //Emergency tickrate, assumes normal tickrate is 64 (as it is with FaceIt Demos)
            return 8;
        }
    }

    /**
     * 
     *
     * @memberof DemoFileParser
     */
    subscribeToDemoEvents() {
        this.SampleRateModulo = this.getTickSampleRateModulo();

        //Beginning of demo file
        this.demoFile.on("start", s => {

            this.ignoreTicks = true; //Ingore ticks until first round starts

            //Put players into an objects that holds both teams
            for (const team of this.demoFile.teams) {
                this.playersInTeams.push({
                    index: team.index,
                    players: team.members,
                });
            }
        });

        //End of the demo file
        this.demoFile.on("end", e => {
            this.on_round_officially_ended();
            console.log(`Parsing of demo ${this.demoFileName} is complete`);
            let millis = Date.now() - this.startTime;
            console.log("seconds elapsed = " + Math.floor(millis / 1000));
        });

        //Get player info at each relevant tick
        this.demoFile.on("tickend", tick => this.on_tickend());

        this.demoFile.gameEvents.on("round_start", s => this.start_ignore_ticks());
        this.demoFile.gameEvents.on("round_freeze_end", f => this.start_parsing_ticks());

		//Write down data at the end of every Round
        this.demoFile.gameEvents.on("round_officially_ended", s => this.on_round_officially_ended());

        console.log("Succesfully subscribed to all events!")
    }

    on_abort() {
        console.log("Parsing of file has been terminated because of irregularities in demo!");
        process.exit(1);
    }

    on_tickend() {
        /** 
         * If this tick is not relevant, do not parse this tick
         */
        if (!this.isRelevantTick()) return;

        if (this.demoFile.currentTick % 2048 == 0) {
            console.log("Current Tick " + this.demoFile.currentTick);
        }

        /** 
         * Get player positions each tick
         */
        this.getAllPlayerInfoForTick();
    }

    /**
     * Is this tick relevant for parsing?
     *
     * @returns
     * @memberof DemoFileParser
     */
    isRelevantTick() {
        return !this.ignoreTicks && (this.demoFile.currentTick % this.SampleRateModulo == 0);
    }

    /**
     * Parses information of all players for current tick
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

        let allPlayers = [].concat(t.members, ct.members);

        const currentTick = this.demoFile.currentTick;

        if (allPlayers.length != 10 || allPlayers.includes(null)) {
            console.log("Current Tick " + currentTick + " | Teams are not complete, discarding this tick!")
            return
        }

        //Info from current tick for all players
        let tickInfo = {
            tick: currentTick
        };

        //Player counter for later loop
        let playerCounter = 0;
        //Go through all players
        for (const player of allPlayers) {

            const featureNameStart = `f_${playerCounter}`

            const playerID = player ? player.userId : "-1";
            const playerName = player ? player.name : "playerMissing";


            let position = player.position;

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

            //Tactical information
            tickInfo[featureNameStart + "_isSpotted"] = player.isSpotted ? 1 : 0;
            tickInfo[featureNameStart + "_isScoped"] = player.isScoped ? 1 : 0;
            tickInfo[featureNameStart + "_isDefusing"] = player.isDefusing ? 1 : 0;

            let allyCounter = 1; //The player itself has index 0.
            let enemyCounter = 0;

            //Distance to other players (Index begins at 1 for allies, 0 for enemies)
            for (const otherPlayer of allPlayers) {

                //Don't measure distance to yourself!
                if (otherPlayer.userId == player.userId) {
                    continue;
                };

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
            console.log("Current Tick " + currentTick + " | Something went wrong, discarding this tick!")
            return
        } else {
            this.playerInfoBuffer = this.playerInfoBuffer.concat(tickInfo);
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

        console.log(`===========> ROUND: ${finishedRound} HAS ENDED!`);

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
        console.log("===========> Data of last Round written to CSV");

        this.playerInfoBuffer.length = 0
        this.deathDataBuffer.length = 0
    }
}

let debug = true
let demoFilePath, parsedFilePath;

if (debug == true) {
    demoFilePath = '../demo_files/sprout-vs-ex-epsilon-m3-overpass.dem';
    parsedFilePath = 'parsed_files/';
} else {
    //TODO different ways to set paths
    demoFilePath = process.argv[2]
    parsedFilePath = process.argv[3]
}

new DemoFileParser(demoFilePath, parsedFilePath);

/*
try {
    fs.unlinkSync('parsed_files\\positions.csv')
    console.log("Deleted old file!")
} catch (e) {
    console.log("Do nothing!")
}
*/

//demo.parseDemoFile(path.resolve('vitality-vs-liquid-m2-dust2.dem'))
